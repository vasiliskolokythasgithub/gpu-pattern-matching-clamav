#include "clamav.h"
#include "matcher-ac.h"
#include "gpu_ac.h"
#include "others.h" 
#define LOCAL_HITS 256
#include <CL/cl.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#define HAVE_OPENCL 1
#define GPU_MAX_HITS 65536
#define LOCAL_SIZE 256
#define MAX_SCAN_SIZE (64 * 1024 * 1024)
#include "matcher.h"
#include "str.h"
#include <ctype.h>
#include "gpu_ac_kernel.h"
#include "gpu_pattern.h"

;

typedef struct {
    cl_event write_done;        /* When data is uploaded */
    cl_event kernel_done;        /* When kernel finishes */
    cl_event count_done;         /* When hit_count is read */
    uint32_t hit_count;           /* Number of hits for this file */
    uint32_t file_idx;             /* Original file index */
    bool processed;                 /* Whether we've collected results */
} batch_file_t;



static int parse_subexpr_with_modifiers(const char *start, const char *end,
                                        gpu_expr_inst_t *bytecode, int max_nodes,
                                        int *pc, int uses_zero_based)
{
    /* This will parse something like "0>5" or "(0|1)>5,2" */
    const char *p = start;
    int modifier = 0;  /* 0=none, 1=>, 2=<, 3== */
    int modval1 = 0, modval2 = 0;
    int subexpr_pc_start = *pc;
    
    /* Skip whitespace if any */
    while (p < end && isspace(*p)) p++;
    
    /* Find where the modifier starts (>, <, =) */
    const char *mod_start = NULL;
    const char *mod_end = NULL;
    const char *comma_pos = NULL;
    
    for (const char *q = p; q < end; q++) {
        if (*q == '>' || *q == '<' || *q == '=') {
            mod_start = q;
            break;
        }
    }
    
    /* If we found a modifier, parse the subexpression before it */
    if (mod_start) {
        /* Parse the left part (the actual expression) */
        int left_len = mod_start - p;
        char *left_expr = cli_malloc(left_len + 1);
        if (!left_expr) return -1;
        strncpy(left_expr, p, left_len);
        left_expr[left_len] = '\0';
        
        /* Recursively parse the left expression */
        int ret = build_lsig_bytecode_internal(left_expr, bytecode, max_nodes, 
                                                pc, uses_zero_based);
        free(left_expr);
        if (ret < 0) return -1;
        
        /* Now parse the modifier */
        const char *mod_str = mod_start;
        if (*mod_str == '>') modifier = 1;
        else if (*mod_str == '<') modifier = 2;
        else if (*mod_str == '=') modifier = 3;
        
        mod_str++;
        
        /* Parse the number(s) after modifier */
        char *endptr;
        modval1 = strtoul(mod_str, &endptr, 10);
        
        /* Check for comma and second value */
        if (*endptr == ',') {
            modval2 = strtoul(endptr + 1, &endptr, 10);
        }
        
        /* Generate the appropriate opcode */
        if (modifier == 1) { /* > */
            bytecode[*pc].op = 7;  /* OP_GT */
            bytecode[*pc].operand = modval1;
            (*pc)++;
            
            if (modval2 > 0) {
                /* For A>X,Y, we need to check both count > X and distinct subsigs >= Y */
                /* This is complex - we'll need to duplicate the result and check mask */
                /* For now, we'll just do a simple implementation */
            }
        } else if (modifier == 2) { /* < */
            bytecode[*pc].op = 8;  /* OP_LT */
            bytecode[*pc].operand = modval1;
            (*pc)++;
        } else if (modifier == 3) { /* = */
            bytecode[*pc].op = 9;  /* OP_EQ */
            bytecode[*pc].operand = modval1;
            (*pc)++;
        }
        
        return 1;
    }
    
    /* No modifier found, just parse as normal expression */
    char *subexpr = cli_malloc(end - start + 1);
    if (!subexpr) return -1;
    strncpy(subexpr, start, end - start);
    subexpr[end - start] = '\0';
    
    int ret = build_lsig_bytecode_internal(subexpr, bytecode, max_nodes, 
                                            pc, uses_zero_based);
    free(subexpr);
    return ret;
}


int build_lsig_bytecode_internal(const char *expr, 
                                        gpu_expr_inst_t *bytecode,
                                        int max_nodes,
                                        int *pc_ptr,
                                        int uses_zero_based)
{
    int pc = *pc_ptr;
    int stack[64];
    int sp = 0;
    int i = 0;
    int len = strlen(expr);
    
    while (i < len && pc < max_nodes - 1) {
        char c = expr[i];
        
        if (isdigit(c)) {
            int subsig = 0;
            while (i < len && isdigit(expr[i])) {
                subsig = subsig * 10 + (expr[i] - '0');
                i++;
            }
            
            /* Check if this subsig has a modifier attached */
            if (i < len && (expr[i] == '>' || expr[i] == '<' || expr[i] == '=')) {
                /* Handle subsig with modifier like "0>5" */
                char mod = expr[i];
                i++;
                
                /* Parse the number after modifier */
                int modval1 = 0, modval2 = 0;
                while (i < len && isdigit(expr[i])) {
                    modval1 = modval1 * 10 + (expr[i] - '0');
                    i++;
                }
                
                /* Check for comma and second value */
                if (i < len && expr[i] == ',') {
                    i++;
                    while (i < len && isdigit(expr[i])) {
                        modval2 = modval2 * 10 + (expr[i] - '0');
                        i++;
                    }
                }
                
                /* Convert subsig to 0-based */
                int operand = uses_zero_based ? subsig : (subsig - 1);
                
                /* First load the count */
                bytecode[pc].op = 5;  /* OP_LOAD_COUNT */
                bytecode[pc].operand = operand;
                pc++;
                
                /* Then apply the comparison */
                if (mod == '>') {
                    bytecode[pc].op = 7;  /* OP_GT */
                    bytecode[pc].operand = modval1;
                    pc++;
                } else if (mod == '<') {
                    bytecode[pc].op = 8;  /* OP_LT */
                    bytecode[pc].operand = modval1;
                    pc++;
                } else if (mod == '=') {
                    bytecode[pc].op = 9;  /* OP_EQ */
                    bytecode[pc].operand = modval1;
                    pc++;
                }
                
                /* If there's a second value (Y condition), we need to check distinct subsigs */
                if (modval2 > 0) {
                    /* This is complex - we'd need to also check the mask */
                    /* For now, we'll just add a placeholder */
                }
            } else {
                /* Regular subsig without modifier */
                int operand = uses_zero_based ? subsig : (subsig - 1);
                bytecode[pc].op = 0;  /* OP_LOAD_SUBSIG */
                bytecode[pc].operand = operand;
                pc++;
            }
            continue;
        }
        
        switch (c) {
            case '&':
                while (sp > 0 && (stack[sp-1] == '&' || stack[sp-1] == '!')) {
                    char op = stack[--sp];
                    bytecode[pc].op = (op == '&') ? 1 : 3;
                    bytecode[pc].operand = 0;
                    pc++;
                }
                stack[sp++] = '&';
                i++;
                break;

            case '|':
                while (sp > 0 && (stack[sp-1] == '&' || stack[sp-1] == '|' || stack[sp-1] == '!')) {
                    char op = stack[--sp];
                    if (op == '&') bytecode[pc].op = 1;
                    else if (op == '|') bytecode[pc].op = 2;
                    else if (op == '!') bytecode[pc].op = 3;
                    bytecode[pc].operand = 0;
                    pc++;
                }
                stack[sp++] = '|';
                i++;
                break;
                
            case '!':
                stack[sp++] = '!';
                i++;
                break;
                
            case '(':
                stack[sp++] = '(';
                i++;
                break;
                
            case ')':
                while (sp > 0 && stack[sp-1] != '(') {
                    char op = stack[--sp];
                    if (op == '&') {
                        bytecode[pc].op = 1;  /* AND */
                    } else if (op == '|') {
                        bytecode[pc].op = 2;  /* OR */
                    } else if (op == '!') {
                        bytecode[pc].op = 3;  /* NOT */
                    }
                    bytecode[pc].operand = 0;
                    pc++;
                }
                if (sp > 0 && stack[sp-1] == '(') {
                    sp--;  /* Pop '(' */
                }
                i++;
                break;
                
            default:
                i++;
                break;
        }
    }
    
    /* Pop remaining operators */
    while (sp > 0 && pc < max_nodes - 1) {
        char op = stack[--sp];
        if (op == '&') {
            bytecode[pc].op = 1;  /* AND */
        } else if (op == '|') {
            bytecode[pc].op = 2;  /* OR */
        } else if (op == '!') {
            bytecode[pc].op = 3;  /* NOT */
        }
        bytecode[pc].operand = 0;
        pc++;
    }
    
    *pc_ptr = pc;
    return 0;
}

 int build_lsig_bytecode(const char *expr, 
                                gpu_expr_inst_t *bytecode,
                                int max_nodes,
                                int *uses_zero_based)
{
    int pc = 0;
    
    /* First pass: detect if expression uses 0-based indices */
    *uses_zero_based = 0;
    const char *p = expr;
    while (*p) {
        if (isdigit(*p)) {
            int subsig = 0;
            while (isdigit(*p)) {
                subsig = subsig * 10 + (*p - '0');
                p++;
            }
            if (subsig == 0) {
                *uses_zero_based = 1;
                break;
            }
        } else {
            p++;
        }
    }
    
    /* Parse the expression recursively */
    int ret = build_lsig_bytecode_internal(expr, bytecode, max_nodes, &pc, *uses_zero_based);
    if (ret < 0) return ret;
    
    /* Add END instruction */
    if (pc < max_nodes - 1) {
        bytecode[pc].op = 4;  /* END */
        bytecode[pc].operand = 0;
        pc++;
    }
    
    return pc;
}



static int get_expr_type(const char *expr)
{
    int has_paren = 0;
    int has_and = 0;
    int has_or = 0;
    
    for (int i = 0; expr[i]; i++) {
        if (expr[i] == '(') has_paren = 1;
        if (expr[i] == '&') has_and = 1;
        if (expr[i] == '|') has_or = 1;
    }
    
    if (!has_paren && has_and && !has_or) return 1;  /* Simple AND */
    if (!has_paren && !has_and && has_or) return 2;  /* Simple OR */
    return 3;  /* Complex (nested or mixed) */
}
 

/**
 * Upload logical signature metadata to GPU
 * Call this after gpu_upload_pattern_metadata
 */

 int gpu_upload_logical_signatures(struct gpu_rt *rt,
                                   struct cli_matcher *root)
{
    cl_int err;


       fprintf(stderr, "=== GPU LSIG UPLOAD ===\n");
    fprintf(stderr, "root=%p\n", (void*)root);
    
    if (!rt || !root) {
        fprintf(stderr, "GPU LSIG: rt or root is NULL\n");
        return -1;
    }
    
    fprintf(stderr, "root->ac_lsigtable=%p\n", (void*)root->ac_lsigtable);
    fprintf(stderr, "root->ac_lsigs=%u\n", root->ac_lsigs);
    
    if (!rt || !root || !root->ac_lsigtable) {
        fprintf(stderr, "GPU LSIG: No logical signatures (null)\n");
        rt->num_lsigs = 0;
        return 0;
    }
    
    /* Count logical signatures */
    uint32_t num_lsigs = 0;
    while (num_lsigs < root->ac_lsigs && 
           root->ac_lsigtable[num_lsigs] != NULL) {
        num_lsigs++;
    }
    
    fprintf(stderr, "GPU LSIG: Found %u logical signatures out of %u\n", 
            num_lsigs, root->ac_lsigs);
    
    if (num_lsigs == 0) {
        rt->num_lsigs = 0;
        return 0;
    }
    
    /* First pass: calculate total bytecode size - we need a better estimate now */
    uint32_t total_expr_nodes = num_lsigs * 64; /* Overestimate to be safe */
    
    /* Allocate host arrays */
    gpu_lsig_meta_t *h_metas = calloc(num_lsigs, sizeof(gpu_lsig_meta_t));
    if (!h_metas) return -1;
    
    gpu_expr_inst_t *h_bytecode = malloc(total_expr_nodes * sizeof(gpu_expr_inst_t));
    if (!h_bytecode) {
        free(h_metas);
        return -1;
    }
    
    uint32_t current_pos = 0;
    for (uint32_t i = 0; i < num_lsigs; i++) {
        struct cli_ac_lsig *lsig = root->ac_lsigtable[i];
        if (!lsig) continue;
        
        gpu_lsig_meta_t *meta = &h_metas[i];
        
        meta->sig_id = i;  
        meta->num_subsigs = lsig->tdb.subsigs;
        meta->expr_offset = current_pos;
        
        /* Handle virus name */
        if (lsig->virname) {
            meta->virname_offset = rt->virname_pool_size;
            meta->virname_len = strlen(lsig->virname);
            
            // fprintf(stderr, "LSIG[%u]: virname='%s', subsigs=%u, bc_idx=%u\n", 
            //         i, lsig->virname, lsig->tdb.subsigs, lsig->bc_idx);
            
            char *new_pool = realloc(rt->h_virname_pool, rt->virname_pool_size + meta->virname_len + 1);
            if (!new_pool) return -1;
            rt->h_virname_pool = new_pool;
            
            strcpy(rt->h_virname_pool + rt->virname_pool_size, lsig->virname);
            rt->virname_pool_size += meta->virname_len + 1;
        } else {
            meta->virname_offset = 0;
            meta->virname_len = 0;
        }
        
        /* TDB fields - populate from lsig->tdb */
        meta->tdb_container = lsig->tdb.container ? lsig->tdb.container[0] : 0;
        meta->tdb_filesize_min = lsig->tdb.filesize ? lsig->tdb.filesize[0] : 0;
        meta->tdb_filesize_max = lsig->tdb.filesize ? lsig->tdb.filesize[1] : 0;
        meta->tdb_ep_min = lsig->tdb.ep ? lsig->tdb.ep[0] : 0;
        meta->tdb_ep_max = lsig->tdb.ep ? lsig->tdb.ep[1] : 0;
        meta->tdb_nos_min = lsig->tdb.nos ? lsig->tdb.nos[0] : 0;
        meta->tdb_nos_max = lsig->tdb.nos ? lsig->tdb.nos[1] : 0;
        meta->tdb_handlertype = lsig->tdb.handlertype ? lsig->tdb.handlertype[0] : 0;
        meta->bc_idx = lsig->bc_idx;
        
        /* Compute intermediates mask */
        meta->tdb_intermediates_mask = 0;
        if (lsig->tdb.intermediates) {
            for (uint32_t j = 1; j <= lsig->tdb.intermediates[0] && j < 32; j++) {
                if (lsig->tdb.intermediates[j] < 32) {
                    meta->tdb_intermediates_mask |= (1u << lsig->tdb.intermediates[j]);
                }
            }
        }
        
        /* Icon groups - would need to be set from somewhere */
        meta->tdb_icongrp1_offset = 0;
        meta->tdb_icongrp2_offset = 0;

           meta->has_regex = lsig->has_regex;
        // if (meta->has_regex) {
        //     fprintf(stderr, "LSIG[%u]: has regex in subsigs\n", i);
        // }
                    
        /* Generate bytecode */
        uint32_t start_pos = current_pos;
        int bytecode_generated = 0;
        int uses_zero_based = 0;

        if (lsig->u.logic) {
            int bytecode_len = build_lsig_bytecode(lsig->u.logic, &h_bytecode[current_pos], 256, &uses_zero_based);
            if (bytecode_len > 0) {
                current_pos += bytecode_len;
                meta->expr_length = bytecode_len;
                bytecode_generated = 1;
                // fprintf(stderr, "LSIG[%u]: compiled expr '%s' -> %d nodes (uses_zero_based=%d)\n", 
                //         i, lsig->u.logic, bytecode_len, uses_zero_based);
            }
        }

        /* Only use fallback if we didn't generate bytecode from expression */
        if (!bytecode_generated) {
            if (lsig->tdb.subsigs == 0) {
                h_bytecode[current_pos].op = 0;
                h_bytecode[current_pos].operand = 0;
                current_pos++;
                h_bytecode[current_pos].op = 3; /* NOT */
                h_bytecode[current_pos].operand = 0;
                current_pos++;
            } else {
                for (uint32_t s = 0; s < lsig->tdb.subsigs; s++) {
                    h_bytecode[current_pos].op = 0;  /* LOAD */
                    h_bytecode[current_pos].operand = s;
                    current_pos++;
                }
                for (uint32_t s = 1; s < lsig->tdb.subsigs; s++) {
                    h_bytecode[current_pos].op = 1; /* AND */
                    h_bytecode[current_pos].operand = 0;
                    current_pos++;
                }
            }
            
            h_bytecode[current_pos].op = 4; /* END */
            h_bytecode[current_pos].operand = 0;
            current_pos++;
            
            meta->expr_length = current_pos - start_pos;
        }
    }
    
    
    /* Upload to GPU */
    rt->d_lsig_metas = clCreateBuffer(rt->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       num_lsigs * sizeof(gpu_lsig_meta_t),
                                       h_metas, &err);
    if (err != CL_SUCCESS) goto error;
    
    rt->d_expr_bytecode = clCreateBuffer(rt->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          current_pos * sizeof(gpu_expr_inst_t),
                                          h_bytecode, &err);
    if (err != CL_SUCCESS) goto error;
    
    rt->num_lsigs = num_lsigs;
    rt->expr_bytecode_size = current_pos;
    
    fprintf(stderr, "GPU LSIG: Uploaded %u signatures, %u expr nodes\n",
            num_lsigs, current_pos);
    

            fprintf(stderr, "=== LOGICAL SIGNATURES IN ROOT ===\n");
for (uint32_t i = 0; i < root->ac_lsigs; i++) {
    struct cli_ac_lsig *lsig = root->ac_lsigtable[i];
    // if (lsig) {
    //     fprintf(stderr, "  LSIG[%u]: sig_id should be %u, virname='%s', subsigs=%u\n",
    //             i, i+1, lsig->virname ? lsig->virname : "NULL", lsig->tdb.subsigs);
    // }
}
    free(h_metas);
    free(h_bytecode);
    return 0;
    
error:
    cli_errmsg("GPU: Failed to upload logical signatures: %d\n", err);
    free(h_metas);
    free(h_bytecode);
    return -1;
}

int gpu_upload_pattern_metadata(struct gpu_rt *rt,
                                 struct cli_matcher *root)
{
        fprintf(stderr, "=== gpu_upload_pattern_metadata START ===\n");
    fprintf(stderr, "rt=%p, root=%p\n", (void*)rt, (void*)root);
    fprintf(stderr, "root->gpu_patt_count=%u\n", root->gpu_patt_count);
    fprintf(stderr, "root->ac_patterns=%u\n", root->ac_patterns);
    fprintf(stderr, "root->gpu_patt_lookup=%p\n", (void*)root->gpu_patt_lookup);
    fprintf(stderr, "gpu_upload_pattern_metadata: Starting...\n");
    fprintf(stderr, "  root=%p, rt=%p, gpu_patt_count=%u\n", 
            (void*)root, (void*)rt, root->gpu_patt_count);

    /* DEBUG: Print all pattern virnames from the matcher */
    // for (uint32_t i = 0; i < root->ac_patterns; i++) {
    //     if (root->ac_pattable[i] && root->ac_pattable[i]->virname) {
    //         fprintf(stderr, "  CPU PATTERN[%u]: virname='%s', sigid=%u, lsigid[0]=%u\n",
    //                 i, root->ac_pattable[i]->virname, 
    //                 root->ac_pattable[i]->sigid,
    //                 root->ac_pattable[i]->lsigid[0]);
    //     }
    // }
    
    cl_int err;
    uint32_t np = root->gpu_patt_count;
    
    if (np == 0) {
        fprintf(stderr, "gpu_upload_pattern_metadata: ERROR - gpu_patt_count is 0\n");
        return -1;
    }
    
    /* Pass 1: Calculate total sizes */
    uint32_t total_pat_bytes = 0;
    uint32_t total_pfx_bytes = 0;
    uint32_t total_vn_bytes = 0;
    
    fprintf(stderr, "  Calculating sizes for %u patterns...\n", np);
    
    for (uint32_t i = 0; i < np; i++) {
        struct cli_ac_patt *p = root->gpu_patt_lookup[i];
        if (!p) {
            fprintf(stderr, "  WARNING: gpu_patt_lookup[%u] is NULL\n", i);
            continue;
        }
        total_pat_bytes += p->length[0];
        total_pfx_bytes += p->prefix_length[0];
        if (p->virname)
            total_vn_bytes += strlen(p->virname) + 1;

        if (p->virname && strstr(p->virname, "Email.Phishing.VOF1-6295567-1")) {
    fprintf(stderr, "FOUND PROBLEM PATTERN: pattern %u, lsig_id=%u, subsig=%u, has_regex=%d\n",
            i, p->lsigid[1], p->lsigid[2], p->has_regex);
}
        
        if (i < 5) {  /* Print first few for debugging */
            fprintf(stderr, "    pattern[%u]: len=%u, pfx_len=%u, virname=%s\n", 
                    i, p->length[0], p->prefix_length[0], 
                    p->virname ? p->virname : "NULL");
        }
    }
    
    fprintf(stderr, "  total_pat_bytes=%u, total_pfx_bytes=%u, total_vn_bytes=%u\n",
            total_pat_bytes, total_pfx_bytes, total_vn_bytes);
    
    /* Ensure minimums for buffer creation */
    if (total_pat_bytes == 0) {
        fprintf(stderr, "  WARNING: total_pat_bytes is 0, setting to 1\n");
        total_pat_bytes = 1;
    }
    if (total_pfx_bytes == 0) {
        fprintf(stderr, "  WARNING: total_pfx_bytes is 0, setting to 1\n");
        total_pfx_bytes = 1;
    }
    if (total_vn_bytes == 0) {
        fprintf(stderr, "  WARNING: total_vn_bytes is 0, setting to 1\n");
        total_vn_bytes = 1;
    }
    
    /* Allocate host arrays */
    fprintf(stderr, "  Allocating host arrays...\n");
    
    gpu_pattern_t *h_patterns = calloc(np, sizeof(gpu_pattern_t));
    uint16_t *h_pat_bytes = malloc(total_pat_bytes * sizeof(uint16_t));
    uint16_t *h_pfx_bytes = malloc(total_pfx_bytes * sizeof(uint16_t));
    char *h_vn_pool = malloc(total_vn_bytes);
    
    fprintf(stderr, "    h_patterns=%p, h_pat_bytes=%p, h_pfx_bytes=%p, h_vn_pool=%p\n",
            (void*)h_patterns, (void*)h_pat_bytes, (void*)h_pfx_bytes, (void*)h_vn_pool);
    
    if (!h_patterns || !h_pat_bytes || !h_pfx_bytes || !h_vn_pool) {
        fprintf(stderr, "  ERROR: Failed to allocate host arrays\n");
        free(h_patterns); free(h_pat_bytes);
        free(h_pfx_bytes); free(h_vn_pool);
        return -1;
    }
    
    /* Pass 2: Fill arrays */
    fprintf(stderr, "  Filling host arrays...\n");
    
    uint32_t pat_off = 0, pfx_off = 0, vn_off = 0;
    
    for (uint32_t i = 0; i < np; i++) {
        struct cli_ac_patt *p = root->gpu_patt_lookup[i];
        if (!p) continue;
        
        gpu_pattern_t *gp = &h_patterns[i];
        uint16_t max_depth = root->ac_maxdepth;
 
        // if (gp->has_regex) {
        //     fprintf(stderr, "GPU UPLOAD: Pattern[%u] has regex flag set, lsig_id=%u, subsig=%u\n", 
        //             i, p->lsigid[1], p->lsigid[2]);
        // }
        
        gp->depth = (p->length[0] < max_depth) ? p->length[0] : max_depth;
        gp->length = p->length[0];
        gp->prefix_length = p->prefix_length[0];
        gp->parts = p->parts;
        gp->type = p->type;
        gp->sigid = p->sigid;
        gp->partno = p->partno;
        gp->offset_min = p->offset_min;
        gp->offset_max = p->offset_max;
        gp->lsigid[0] = p->lsigid[0];
        gp->lsigid[1] = p->lsigid[1];
        gp->boundary = p->boundary;  /* Copy boundary flags from CPU pattern */
        gp->sigopts = p->sigopts;    /* Copy signature options */
        
        /* Get number of subsigs from lsig table if this is a logical signature */
        gp->has_regex = 0;  // Default

            if (p->lsigid[0] > 0) {  // This pattern belongs to a logical signature
                uint32_t lsig_idx = p->lsigid[1];  // Logical signature ID (1-based in lsigid[1])
                if (lsig_idx > 0 && lsig_idx <= root->ac_lsigs) {
                    struct cli_ac_lsig *lsig = root->ac_lsigtable[lsig_idx - 1];
                    if (lsig) {
                        gp->has_regex = lsig->has_regex;  // Copy from the logical signature
                        if (gp->has_regex) {
                            fprintf(stderr, "GPU UPLOAD: Pattern[%u] inherits regex flag from lsig %u\n", 
                                    i, lsig_idx);
                        }
                    }
                }
            }
        gp->mindist = p->mindist;
        gp->maxdist = p->maxdist;
        
        /* Map ClamAV offset types to GPU constants */
        if (p->offdata[0] == CLI_OFF_ANY || p->offdata[0] == 0) {
            gp->offdata0 = GPU_OFF_ANY;
        } else if (p->offdata[0] == CLI_OFF_NONE) {
            gp->offdata0 = GPU_OFF_NONE;
        } else if (p->offdata[0] == CLI_OFF_ABSOLUTE) {
            gp->offdata0 = GPU_OFF_ABSOLUTE;
        } else {
            gp->offdata0 = GPU_OFF_ANY;
        }
        
        gp->ch0 = p->ch[0];
        gp->ch1 = p->ch[1];
        gp->ch_mindist0 = p->ch_mindist[0];
        gp->ch_mindist1 = p->ch_mindist[1];
        gp->ch_maxdist0 = p->ch_maxdist[0];
        gp->ch_maxdist1 = p->ch_maxdist[1];
        
        /* Pattern bytes */
        gp->pattern_offset = pat_off;
        if (p->pattern) {
            for (uint16_t j = 0; j < p->length[0]; j++) {
                if (pat_off >= total_pat_bytes) {
                    fprintf(stderr, "  ERROR: pat_off overflow: %u >= %u\n", 
                            pat_off, total_pat_bytes);
                    goto fail;
                }
                h_pat_bytes[pat_off++] = p->pattern[j];
            }
        }
        
        /* Prefix bytes */
        gp->prefix_offset = pfx_off;
        if (p->prefix) {
            for (uint16_t j = 0; j < p->prefix_length[0]; j++) {
                if (pfx_off >= total_pfx_bytes) {
                    fprintf(stderr, "  ERROR: pfx_off overflow: %u >= %u\n", 
                            pfx_off, total_pfx_bytes);
                    goto fail;
                }
                h_pfx_bytes[pfx_off++] = p->prefix[j];
            }
        }
        
        /* Virname */
        if (p->virname) {
            gp->virname_offset = vn_off;
            gp->virname_len = strlen(p->virname);

            if (p->virname && strstr(p->virname, "TestSig")) {
    fprintf(stderr, "*** UPLOADING TEST SIG: %s, pid=%u, sigid=%u, lsigid[0]=%u\n",
            p->virname, i, p->sigid, p->lsigid[0]);
}
            
            if (vn_off + gp->virname_len + 1 > total_vn_bytes) {
                fprintf(stderr, "  ERROR: vn_off overflow: %u + %u > %u\n",
                        vn_off, gp->virname_len + 1, total_vn_bytes);
                goto fail;
            }
            
            memcpy(h_vn_pool + vn_off, p->virname, gp->virname_len + 1);
            vn_off += gp->virname_len + 1;
        } else {
            gp->virname_offset = 0;
            gp->virname_len = 0;
        }
        
        if (i < 5) {
            fprintf(stderr, "    filled[%u]: depth=%u, len=%u, off=%u\n",
                    i, gp->depth, gp->length, gp->pattern_offset);
        }
    }

    fprintf(stderr, "GPU DEBUG: First 5 patterns:\n");
    for (uint32_t i = 0; i < 5 && i < np; i++) {
        struct cli_ac_patt *p = root->gpu_patt_lookup[i];
        if (!p) continue;
        gpu_pattern_t *gp = &h_patterns[i];
        
        fprintf(stderr, "  Pattern[%u]: sig_id=%u, virname=%s\n", 
                i, p->sigid, p->virname ? p->virname : "NULL");
        fprintf(stderr, "    depth=%u, length=%u, pattern_offset=%u\n",
                gp->depth, gp->length, gp->pattern_offset);
        fprintf(stderr, "    First 8 pattern bytes: ");
        for (int j = 0; j < 8 && j < p->length[0]; j++) {
            fprintf(stderr, "0x%04x ", p->pattern[j]);
        }
        fprintf(stderr, "\n");
    }
    
    fprintf(stderr, "  Final offsets: pat_off=%u/%u, pfx_off=%u/%u, vn_off=%u/%u\n",
            pat_off, total_pat_bytes, pfx_off, total_pfx_bytes, vn_off, total_vn_bytes);
    
    /* Upload to GPU */
    fprintf(stderr, "  Creating GPU buffers...\n");
    
    rt->num_pattern_bytes = pat_off;
    rt->num_prefix_bytes = pfx_off;
    
    rt->d_patterns = clCreateBuffer(rt->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     np * sizeof(gpu_pattern_t), h_patterns, &err);
    fprintf(stderr, "    d_patterns: err=%d\n", err);
    if (err != CL_SUCCESS) goto fail;
    
    rt->d_pattern_bytes = clCreateBuffer(rt->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          pat_off * sizeof(uint16_t), h_pat_bytes, &err);
    fprintf(stderr, "    d_pattern_bytes: err=%d\n", err);
    if (err != CL_SUCCESS) goto fail;
    
    rt->d_prefix_bytes = clCreateBuffer(rt->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         pfx_off * sizeof(uint16_t), h_pfx_bytes, &err);
    fprintf(stderr, "    d_prefix_bytes: err=%d\n", err);
    if (err != CL_SUCCESS) goto fail;
    
  
    fprintf(stderr, "    d_virname_pool: err=%d\n", err);
    if (err != CL_SUCCESS) goto fail;
    
    /* Tracker pool (read-write, zeroed) */
    rt->d_tracker_pool = clCreateBuffer(rt->context, CL_MEM_READ_WRITE,
                                         GPU_MAX_TRACKERS * sizeof(gpu_multipart_tracker_t),
                                         NULL, &err);
    fprintf(stderr, "    d_tracker_pool: err=%d\n", err);
    if (err != CL_SUCCESS) {
        printf("Failed to create tracker pool: %d\n", err);
        goto fail;
    }
    
    rt->d_tracker_count = clCreateBuffer(rt->context, CL_MEM_READ_WRITE,
                                          sizeof(uint32_t), NULL, &err);
    fprintf(stderr, "    d_tracker_count: err=%d\n", err);
    if (err != CL_SUCCESS) goto fail;
    
    /* Result buffer */
    rt->d_result = clCreateBuffer(rt->context, CL_MEM_READ_WRITE,
                                   sizeof(gpu_scan_result_t), NULL, &err);
    fprintf(stderr, "    d_result: err=%d\n", err);
    if (err != CL_SUCCESS) goto fail;
    
    /* Keep host copy of virname pool */
    fprintf(stderr, "  Allocating host virname pool of size %u\n", vn_off);
    rt->h_virname_pool = malloc(vn_off);
    if (!rt->h_virname_pool) {
        fprintf(stderr, "  ERROR: Failed to allocate host virname pool\n");
        goto fail;
    }
    memcpy(rt->h_virname_pool, h_vn_pool, vn_off);
    rt->virname_pool_size = vn_off;
    rt->num_gpu_patterns = np;



     rt->d_virname_pool = clCreateBuffer(rt->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     rt->virname_pool_size, rt->h_virname_pool, &err);
    
    fprintf(stderr, "  Cleaning up host arrays...\n");
    free(h_patterns);
    free(h_pat_bytes);
    free(h_pfx_bytes);
    free(h_vn_pool);
    
    rt->v2_uploaded = 1;
    fprintf(stderr, "gpu_upload_pattern_metadata: SUCCESS\n");
    return 0;
    
fail:
    fprintf(stderr, "gpu_upload_pattern_metadata: FAILED at step above (err=%d)\n", err);
    free(h_patterns);
    free(h_pat_bytes);
    free(h_pfx_bytes);
    free(h_vn_pool);
    return -1;
}
 

 
int gpu_scan(struct gpu_rt *rt,
             const unsigned char *file_buffer,
             uint32_t file_length,
             const char **virname,
             uint32_t maxpatlen,
             cli_ctx *ctx,
             struct cli_target_info *tinfo)
{
    cl_int err; 

    uint32_t container_type = 0;
    uint32_t entry_point = 0;
    uint32_t is_pe_target = 0;
    
    /* Get TDB information from context (like you do for logical kernel) */
    if (tinfo && tinfo->status == 1) {
        entry_point = tinfo->exeinfo.ep;
        is_pe_target = 1;
    }
    
    if (ctx && ctx->recursion_level >= 0) {
        container_type = cli_recursion_stack_get_type(ctx, -1);
    }
    


    if (!rt || !rt->dfa_uploaded || !rt->v2_uploaded) {
        return GPU_RESULT_BREAK;
    }

    if (file_length > rt->scan_buffer_size) {
        fprintf(stderr, "FAILED due NOT FITTING TO BUFFER SIZE\n");
        return GPU_RESULT_BREAK;
    } 
    
    /* Upload file (async) */
    clEnqueueWriteBuffer(rt->queue, rt->scan_buffer, CL_FALSE,
                          0, file_length, file_buffer, 0, NULL, NULL);
   
    /* Reset result to CLEAN */
    gpu_scan_result_t zero_result = {0};
    clEnqueueWriteBuffer(rt->queue, rt->d_result, CL_FALSE,
                          0, sizeof(gpu_scan_result_t), &zero_result, 0, NULL, NULL);

    /* Reset tracker count to 0 */
    uint32_t zero = 0;
    clEnqueueFillBuffer(rt->queue, rt->d_tracker_pool, &zero, sizeof(zero),
                    0, GPU_MAX_TRACKERS * sizeof(gpu_multipart_tracker_t),
                    0, NULL, NULL);
    clEnqueueWriteBuffer(rt->queue, rt->d_tracker_count, CL_FALSE,
                          0, sizeof(uint32_t), &zero, 0, NULL, NULL);

    /* Wait for data upload to complete */
    clFinish(rt->queue);
    clFinish(rt->queue); 
    fflush(stdout);  // Force flush stdout

    /* Calculate launch params */
    uint32_t chunk_size;

    
    if (file_length >= 128 * 1024 * 1024)      chunk_size = 524288;
    else if (file_length >= 64 * 1024 * 1024)   chunk_size = 262144;
    else if (file_length >= 16 * 1024 * 1024)   chunk_size = 131072;
    else if (file_length >= 4 * 1024 * 1024)    chunk_size = 65536;
    else if (file_length >= 1 * 1024 * 1024)    chunk_size = 32768;
    else                                         chunk_size = 16384;

    //fprintf(stderr, "DEBUG: file_length=%u, chunk_size=%u\n", file_length, chunk_size);

    if (maxpatlen < 2) maxpatlen = 2;
    uint32_t file_offset = 0;
    uint32_t stride = chunk_size - (maxpatlen - 1);
    uint32_t chunks = (file_length + stride - 1) / stride;
    size_t gws = chunks;  // One work item per chunk
    if (gws == 0) gws = 1;
    gws = ((gws + 255) / 256) * 256;  // Round up to workgroup size
    uint32_t max_trackers = GPU_MAX_TRACKERS;

    // fprintf(stderr, "DEBUG: max_trackers=%u, GPU_MAX_TRACKERS=%u\n", 
    //     max_trackers, GPU_MAX_TRACKERS);

    // fprintf(stderr, "DEBUG: stride=%u, chunks=%u, gws=%zu\n", stride, chunks, gws);

    uint32_t is_elf_target = 0;
    if (file_length >= 4 && file_buffer[0] == 0x7F && 
        file_buffer[1] == 'E' && file_buffer[2] == 'L' && file_buffer[3] == 'F') {
        is_elf_target = 1;
    }

    /* Set kernel args for MAIN kernel */
   int a = 0; 

   //fprintf(stderr, "=== SETTING MAIN KERNEL ARGS ===\n");

    err = clSetKernelArg(rt->kernel, a++, sizeof(cl_mem), &rt->scan_buffer);
  //  fprintf(stderr, "Arg %d: scan_buffer=%p, err=%d\n", a-1, (void*)rt->scan_buffer, err);
    if (err != CL_SUCCESS) fprintf(stderr, "ERROR: Failed to set scan_buffer arg\n");

    err = clSetKernelArg(rt->kernel, a++, sizeof(uint32_t), &file_offset);
   // fprintf(stderr, "Arg %d: file_offset=%u, err=%d\n", a-1, file_offset, err);
    if (err != CL_SUCCESS) fprintf(stderr, "ERROR: Failed to set file_offset arg\n");

    err = clSetKernelArg(rt->kernel, a++, sizeof(uint32_t), &file_length);
   // fprintf(stderr, "Arg %d: file_length=%u, err=%d\n", a-1, file_length, err);
    if (err != CL_SUCCESS) fprintf(stderr, "ERROR: Failed to set file_length arg\n");

    err = clSetKernelArg(rt->kernel, a++, sizeof(cl_mem), &rt->dfa_next);
   // fprintf(stderr, "Arg %d: dfa_next=%p, err=%d\n", a-1, (void*)rt->dfa_next, err);
    if (err != CL_SUCCESS) fprintf(stderr, "ERROR: Failed to set dfa_next arg\n");
    if (!rt->dfa_next) fprintf(stderr, "ERROR: dfa_next is NULL!\n");

    err = clSetKernelArg(rt->kernel, a++, sizeof(cl_mem), &rt->dfa_out_index);
   // fprintf(stderr, "Arg %d: dfa_out_index=%p, err=%d\n", a-1, (void*)rt->dfa_out_index, err);
    if (err != CL_SUCCESS) fprintf(stderr, "ERROR: Failed to set dfa_out_index arg\n");
    if (!rt->dfa_out_index) fprintf(stderr, "ERROR: dfa_out_index is NULL!\n");

    err = clSetKernelArg(rt->kernel, a++, sizeof(cl_mem), &rt->dfa_out_count);
   // fprintf(stderr, "Arg %d: dfa_out_count=%p, err=%d\n", a-1, (void*)rt->dfa_out_count, err);
    if (err != CL_SUCCESS) fprintf(stderr, "ERROR: Failed to set dfa_out_count arg\n");
    if (!rt->dfa_out_count) fprintf(stderr, "ERROR: dfa_out_count is NULL!\n");

    err = clSetKernelArg(rt->kernel, a++, sizeof(cl_mem), &rt->dfa_out_pat);
    //fprintf(stderr, "Arg %d: dfa_out_pat=%p, err=%d\n", a-1, (void*)rt->dfa_out_pat, err);
    if (err != CL_SUCCESS) fprintf(stderr, "ERROR: Failed to set dfa_out_pat arg\n");
    if (!rt->dfa_out_pat) fprintf(stderr, "ERROR: dfa_out_pat is NULL!\n");

    err = clSetKernelArg(rt->kernel, a++, sizeof(uint32_t), &rt->dfa_states);
  //  fprintf(stderr, "Arg %d: dfa_states=%u, err=%d\n", a-1, rt->dfa_states, err);
    if (err != CL_SUCCESS) fprintf(stderr, "ERROR: Failed to set dfa_states arg\n");

    err = clSetKernelArg(rt->kernel, a++, sizeof(cl_mem), &rt->d_patterns);
    //fprintf(stderr, "Arg %d: d_patterns=%p, err=%d\n", a-1, (void*)rt->d_patterns, err);
    if (err != CL_SUCCESS) fprintf(stderr, "ERROR: Failed to set d_patterns arg\n");
    if (!rt->d_patterns) fprintf(stderr, "ERROR: d_patterns is NULL!\n");

    err = clSetKernelArg(rt->kernel, a++, sizeof(uint32_t), &rt->num_gpu_patterns);
   // fprintf(stderr, "Arg %d: num_gpu_patterns=%u, err=%d\n", a-1, rt->num_gpu_patterns, err);
    if (err != CL_SUCCESS) fprintf(stderr, "ERROR: Failed to set num_gpu_patterns arg\n");

    err = clSetKernelArg(rt->kernel, a++, sizeof(cl_mem), &rt->d_pattern_bytes);
   // fprintf(stderr, "Arg %d: d_pattern_bytes=%p, err=%d\n", a-1, (void*)rt->d_pattern_bytes, err);
    if (err != CL_SUCCESS) fprintf(stderr, "ERROR: Failed to set d_pattern_bytes arg\n");
    if (!rt->d_pattern_bytes) fprintf(stderr, "ERROR: d_pattern_bytes is NULL!\n");

    err = clSetKernelArg(rt->kernel, a++, sizeof(cl_mem), &rt->d_prefix_bytes);
 //   fprintf(stderr, "Arg %d: d_prefix_bytes=%p, err=%d\n", a-1, (void*)rt->d_prefix_bytes, err);
    if (err != CL_SUCCESS) fprintf(stderr, "ERROR: Failed to set d_prefix_bytes arg\n");
    if (!rt->d_prefix_bytes) fprintf(stderr, "ERROR: d_prefix_bytes is NULL!\n");

    err = clSetKernelArg(rt->kernel, a++, sizeof(cl_mem), &rt->d_tracker_pool);
  //  fprintf(stderr, "Arg %d: d_tracker_pool=%p, err=%d\n", a-1, (void*)rt->d_tracker_pool, err);
    if (err != CL_SUCCESS) fprintf(stderr, "ERROR: Failed to set d_tracker_pool arg\n");
    if (!rt->d_tracker_pool) fprintf(stderr, "ERROR: d_tracker_pool is NULL!\n");

    err = clSetKernelArg(rt->kernel, a++, sizeof(cl_mem), &rt->d_tracker_count);
  //  fprintf(stderr, "Arg %d: d_tracker_count=%p, err=%d\n", a-1, (void*)rt->d_tracker_count, err);
    if (err != CL_SUCCESS) fprintf(stderr, "ERROR: Failed to set d_tracker_count arg\n");
    if (!rt->d_tracker_count) fprintf(stderr, "ERROR: d_tracker_count is NULL!\n");

    err = clSetKernelArg(rt->kernel, a++, sizeof(uint32_t), &max_trackers);
  //  fprintf(stderr, "Arg %d: max_trackers=%u, err=%d\n", a-1, max_trackers, err);
    if (err != CL_SUCCESS) fprintf(stderr, "ERROR: Failed to set max_trackers arg\n");

    err = clSetKernelArg(rt->kernel, a++, sizeof(uint32_t), &chunk_size);
   // fprintf(stderr, "Arg %d: chunk_size=%u, err=%d\n", a-1, chunk_size, err);
    if (err != CL_SUCCESS) fprintf(stderr, "ERROR: Failed to set chunk_size arg\n");

    err = clSetKernelArg(rt->kernel, a++, sizeof(uint32_t), &stride);
   // fprintf(stderr, "Arg %d: stride=%u, err=%d\n", a-1, stride, err);
    if (err != CL_SUCCESS) fprintf(stderr, "ERROR: Failed to set stride arg\n");

    /* Arg 17 is local memory, no check needed */
    err = clSetKernelArg(rt->kernel, a++, 256 * sizeof(uint32_t), NULL);
   // fprintf(stderr, "Arg %d: local memory, err=%d\n", a-1, err);
    if (err != CL_SUCCESS) fprintf(stderr, "ERROR: Failed to set local memory arg\n");

    err = clSetKernelArg(rt->kernel, a++, sizeof(cl_mem), &rt->d_result);
  //  fprintf(stderr, "Arg %d: d_result=%p, err=%d\n", a-1, (void*)rt->d_result, err);
    if (err != CL_SUCCESS) fprintf(stderr, "ERROR: Failed to set d_result arg\n");
    if (!rt->d_result) fprintf(stderr, "ERROR: d_result is NULL!\n");

    err = clSetKernelArg(rt->kernel, a++, sizeof(uint32_t), &rt->num_pattern_bytes);
  //  fprintf(stderr, "Arg %d: num_pattern_bytes=%u, err=%d\n", a-1, rt->num_pattern_bytes, err);
    if (err != CL_SUCCESS) fprintf(stderr, "ERROR: Failed to set num_pattern_bytes arg\n");

    err = clSetKernelArg(rt->kernel, a++, sizeof(uint32_t), &rt->num_prefix_bytes);
   // fprintf(stderr, "Arg %d: num_prefix_bytes=%u, err=%d\n", a-1, rt->num_prefix_bytes, err);
    if (err != CL_SUCCESS) fprintf(stderr, "ERROR: Failed to set num_prefix_bytes arg\n");

    err = clSetKernelArg(rt->kernel, a++, sizeof(uint32_t), &rt->out_total);
   // fprintf(stderr, "Arg %d: out_total=%u, err=%d\n", a-1, rt->out_total, err);
    if (err != CL_SUCCESS) fprintf(stderr, "ERROR: Failed to set out_total arg\n");

    


    err = clSetKernelArg(rt->kernel, a++, sizeof(cl_mem), &rt->d_lsig_metas);
    err = clSetKernelArg(rt->kernel, a++, sizeof(uint32_t), &rt->num_lsigs);
    err = clSetKernelArg(rt->kernel, a++, sizeof(uint32_t), &container_type);
    err = clSetKernelArg(rt->kernel, a++, sizeof(uint32_t), &entry_point);
    err = clSetKernelArg(rt->kernel, a++, sizeof(uint32_t), &is_pe_target);
    err = clSetKernelArg(rt->kernel, a++, sizeof(uint32_t), &is_elf_target);

    
    fprintf(stderr, "Total args set: %d (expected 22)\n", a); 
    fflush(stderr);

    /* Launch MAIN kernel */
err = clEnqueueNDRangeKernel(rt->queue, rt->kernel, 1, NULL,
                              &gws, NULL, 0, NULL, NULL);
if (err != CL_SUCCESS) {
    fprintf(stderr, "GPU: Main kernel launch failed: %d\n", err);
    return GPU_RESULT_BREAK;
}

/* Force completion and flush */
clFinish(rt->queue);
clFinish(rt->queue);  // Do it twice
fflush(stdout); 

/* If we have logical signatures, run evaluation kernel */
/* If we have logical signatures, run evaluation kernel */
/* If we have logical signatures, run evaluation kernel */
if (rt->num_lsigs > 0 && rt->lsig_kernel) {
    /* Get tracker count from main kernel */
    uint32_t tracker_count;
    clEnqueueReadBuffer(rt->queue, rt->d_tracker_count, CL_TRUE,
                        0, sizeof(uint32_t), &tracker_count, 0, NULL, NULL);
    
    /* Get TDB information from context */
    uint32_t file_length_arg = file_length;
    uint32_t entry_point = (tinfo && tinfo->status == 1) ? tinfo->exeinfo.ep : 0;
    uint32_t num_sections = (tinfo && tinfo->status == 1) ? tinfo->exeinfo.nsections : 0;
    uint32_t container_type = (ctx && ctx->recursion_level >= 0) ? 
                            cli_recursion_stack_get_type(ctx, -1) : 0;
    uint32_t num_intermediates = 0;
    uint32_t is_pe_target = (tinfo && tinfo->status == 1) ? 1 : 0;
    uint32_t expr_bytecode_size = rt->expr_bytecode_size;  /* Make sure this is stored in rt struct */

    /* Get intermediates (parent types) */
    uint32_t intermediates[8] = {0};
    if (ctx && ctx->recursion_level >= 0) {
        int32_t j = -2;
        while (j >= -((int32_t)ctx->recursion_level) && num_intermediates < 8) {
            uint32_t type = cli_recursion_stack_get_type(ctx, j);
            if (type != CL_TYPE_ANY) {
                intermediates[num_intermediates++] = type;
            }
            j--;
        }
    }
    
    /* Create intermediates buffer */
    cl_mem d_intermediates = NULL;
    if (num_intermediates > 0) {
        d_intermediates = clCreateBuffer(rt->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         num_intermediates * sizeof(uint32_t), intermediates, NULL);
    } else {
        uint32_t dummy = 0;
        d_intermediates = clCreateBuffer(rt->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         sizeof(uint32_t), &dummy, NULL);
    }
    
    /* Create dummy icon buffer */
    uint32_t dummy_icon = 0;
    cl_mem d_icon_data = clCreateBuffer(rt->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         sizeof(uint32_t), &dummy_icon, NULL);
    uint32_t icon_data_size = 0;
    
    /* Set args for logical signature kernel */
    int a = 0;
    clSetKernelArg(rt->lsig_kernel, a++, sizeof(cl_mem), &rt->d_lsig_metas);
    clSetKernelArg(rt->lsig_kernel, a++, sizeof(uint32_t), &rt->num_lsigs);
    clSetKernelArg(rt->lsig_kernel, a++, sizeof(cl_mem), &rt->d_expr_bytecode);
    clSetKernelArg(rt->lsig_kernel, a++, sizeof(uint32_t), &expr_bytecode_size);  /* NEW */
    clSetKernelArg(rt->lsig_kernel, a++, sizeof(cl_mem), &rt->d_tracker_pool);
    clSetKernelArg(rt->lsig_kernel, a++, sizeof(uint32_t), &tracker_count);
    clSetKernelArg(rt->lsig_kernel, a++, sizeof(uint32_t), &file_length_arg);
    clSetKernelArg(rt->lsig_kernel, a++, sizeof(uint32_t), &entry_point);
    clSetKernelArg(rt->lsig_kernel, a++, sizeof(uint32_t), &num_sections);
    clSetKernelArg(rt->lsig_kernel, a++, sizeof(uint32_t), &container_type);
    clSetKernelArg(rt->lsig_kernel, a++, sizeof(cl_mem), &d_intermediates);
    clSetKernelArg(rt->lsig_kernel, a++, sizeof(uint32_t), &num_intermediates);
    clSetKernelArg(rt->lsig_kernel, a++, sizeof(uint32_t), &is_pe_target);
    clSetKernelArg(rt->lsig_kernel, a++, sizeof(cl_mem), &d_icon_data);
    clSetKernelArg(rt->lsig_kernel, a++, sizeof(uint32_t), &icon_data_size);
    clSetKernelArg(rt->lsig_kernel, a++, sizeof(cl_mem), &rt->d_result);
    clSetKernelArg(rt->lsig_kernel, a++, sizeof(cl_mem), &rt->d_virname_pool);
    
     
    
    size_t gws = ((rt->num_lsigs + 255) / 256) * 256;
    clEnqueueNDRangeKernel(rt->queue, rt->lsig_kernel, 1, NULL,
                        &gws, NULL, 0, NULL, NULL);
    
    /* Wait for kernel to complete before cleaning up */
    clFinish(rt->queue);
    fflush(stdout); 
    
     
    if (d_intermediates) clReleaseMemObject(d_intermediates);
    if (d_icon_data) clReleaseMemObject(d_icon_data);
}

    clFinish(rt->queue); 
        clFinish(rt->queue);
        fflush(stdout); 
    /* Read final result */
    gpu_scan_result_t h_result;
    clEnqueueReadBuffer(rt->queue, rt->d_result, CL_TRUE,
                        0, sizeof(gpu_scan_result_t), &h_result, 0, NULL, NULL);

                        fprintf(stderr, "GPU RESULT: result_code=%d, virname_offset=%u, virname_len=%u\n",
        h_result.result_code, h_result.virname_offset, h_result.virname_len);


        if (h_result.needs_cpu_fallback) {
    fprintf(stderr, "GPU: Falling back to CPU due to regex pattern at offset %u\n", 
            h_result.fallback_offset);
    return GPU_RESULT_BREAK;  /* This will trigger CPU fallback */
}

if (h_result.result_code == GPU_RESULT_VIRUS && h_result.virname_offset < rt->virname_pool_size) {
    fprintf(stderr, "Virus name from pool: '%s'\n", 
            rt->h_virname_pool + h_result.virname_offset);
}
     
    
    if (h_result.result_code == GPU_RESULT_VIRUS) {
        *virname = rt->h_virname_pool + h_result.virname_offset;
        return GPU_RESULT_VIRUS;
    }
    
    return h_result.result_code;
}


/**
 * Convert logical expression string to bytecode
 * Input: "1&2&3" or "(1&2)|(3&4)"
 */
static int compile_logical_expr(const char *expr, 
                                 gpu_expr_inst_t *bytecode,
                                 int max_len) {
    /* Use shunting-yard algorithm to convert infix to postfix */
    int pc = 0;
    int op_stack[64];
    int sp = 0;
    
    for (int i = 0; expr[i] && pc < max_len; i++) {
        if (isdigit(expr[i])) {
            /* Parse subsig number */
            int subsig = 0;
            while (isdigit(expr[i])) {
                subsig = subsig * 10 + (expr[i] - '0');
                i++;
            }
            i--; /* Adjust for loop increment */
            
            bytecode[pc].op = OP_LOAD_SUBSIG;
            bytecode[pc].operand = subsig - 1; /* 0-based */
            pc++;
        }
        else switch (expr[i]) {
            case '&':
                while (sp > 0 && op_stack[sp-1] != '(') {
                    bytecode[pc].op = OP_AND;
                    bytecode[pc].operand = 0;
                    pc++;
                    sp--;
                }
                op_stack[sp++] = '&';
                break;
                
            case '|':
                while (sp > 0 && op_stack[sp-1] != '(') {
                    bytecode[pc].op = OP_OR;
                    bytecode[pc].operand = 0;
                    pc++;
                    sp--;
                }
                op_stack[sp++] = '|';
                break;
                
            case '!':
                op_stack[sp++] = '!';
                break;
                
            case '(':
                op_stack[sp++] = '(';
                break;
                
            case ')':
                while (sp > 0 && op_stack[sp-1] != '(') {
                    int op = op_stack[--sp];
                    if (op == '&') {
                        bytecode[pc].op = OP_AND;
                    } else if (op == '|') {
                        bytecode[pc].op = OP_OR;
                    } else if (op == '!') {
                        bytecode[pc].op = OP_NOT;
                    }
                    bytecode[pc].operand = 0;
                    pc++;
                }
                if (sp > 0 && op_stack[sp-1] == '(') {
                    sp--; /* Pop '(' */
                }
                break;
        }
    }
    
    /* Pop remaining operators */
    while (sp > 0 && pc < max_len) {
        int op = op_stack[--sp];
        if (op == '&') {
            bytecode[pc].op = OP_AND;
        } else if (op == '|') {
            bytecode[pc].op = OP_OR;
        } else if (op == '!') {
            bytecode[pc].op = OP_NOT;
        }
        bytecode[pc].operand = 0;
        pc++;
    }
    
    bytecode[pc].op = OP_END;
    bytecode[pc].operand = 0;
    pc++;
    
    return pc;
}

 

cl_error_t gpu_collect_batch_results(struct gpu_rt *rt, struct cli_matcher *root,
                                      cli_ctx *ctx, const char **virname)
{
    if (!rt->batch.active) return CL_SUCCESS;
    
    cl_error_t final_result = CL_CLEAN;
    
    for (uint32_t f = 0; f < rt->batch.count; f++) {
        if (rt->batch.results[f] == CL_VIRUS) {
            if (virname && rt->batch.virnames[f])
                *virname = rt->batch.virnames[f];
            final_result = CL_VIRUS;
            break;
        }
        if (rt->batch.results[f] == CL_BREAK) {
            final_result = CL_BREAK;
        }
    }
    
    /* Cleanup */
    for (uint32_t i = 0; i < rt->batch.count; i++) {
        if (rt->batch.buffers[i]) {
            free((void*)rt->batch.buffers[i]);
            rt->batch.buffers[i] = NULL;
        }
    }
    
    free(rt->batch.hit_counts); rt->batch.hit_counts = NULL;
    free(rt->batch.results); rt->batch.results = NULL;
    free(rt->batch.virnames); rt->batch.virnames = NULL;
    rt->batch.count = 0;
    rt->batch.total_bytes = 0;
    rt->batch.active = false;
    
    return final_result;
}

 

 

 


/* =========================
 * Upload DFA
 * ========================= */
/* ==========================================
 * OPTIMIZED DFA UPLOAD TO GPU
 * Key improvements:
 * 1. Use pinned memory for faster transfers
 * 2. Async transfers with events
 * 3. Compress sparse outputs
 * ========================================== */

/* ============================================
 * KEY OPTIMIZATION #5: One-time upload on init
 * Upload DFA once during engine load, not per-scan
 * ============================================ */

int gpu_rt_init(struct gpu_rt *rt)
{
    cl_int err;
    
    if (!rt) return -1;
    if (rt->initialized) return 0;
    
    /* Get OpenCL platform */
    if (clGetPlatformIDs(1, &rt->platform, NULL) != CL_SUCCESS)
        return -1;

    /* Get GPU device */
    if (clGetDeviceIDs(rt->platform, CL_DEVICE_TYPE_GPU,
                       1, &rt->device, NULL) != CL_SUCCESS)
        return -1;

    /* Create OpenCL context */
    rt->context = clCreateContext(NULL, 1, &rt->device, NULL, NULL, &err);
    if (!rt->context)
        return -1;

    /* Create command queue */
    rt->queue = clCreateCommandQueue(rt->context, rt->device, 0, &err);
    if (!rt->queue)
        return -1;

    /* Try flattened kernel first */
    size_t kernel_len = strlen(gpu_kernel_src);
    rt->program = clCreateProgramWithSource(
        rt->context, 1, &gpu_kernel_src, &kernel_len, &err
    );

       if (err != CL_SUCCESS) {
        cli_errmsg("GPU: Failed to create batch buffer (err=%d)\n", err);
        return -1;
    }
    cli_dbgmsg("GPU: Created batch buffer of %zu bytes\n", rt->scan_buffer_size);

 /* Initialize batch processing */
      rt->batch.max_count = 64;
    rt->batch.buffers = malloc(rt->batch.max_count * sizeof(void*));
    rt->batch.lengths = malloc(rt->batch.max_count * sizeof(uint32_t));
    rt->batch.file_offsets = malloc(rt->batch.max_count * sizeof(uint32_t));
    rt->scan_buffer_size = 256 * 1024 * 1024; /* 256MB */
    
    rt->scan_buffer = clCreateBuffer(rt->context, CL_MEM_READ_WRITE,
                                             rt->scan_buffer_size,
                                             NULL, &err);
    if (err != CL_SUCCESS) {
        cli_errmsg("GPU: Failed to create batch buffer\n");
        return -1;
    }
    
    rt->batch.count = 0;
    rt->batch.total_bytes = 0;
    rt->batch.active = false;
    
    /* Allocate event arrays (will be realloc'd as needed) */
    rt->batch.kernel_events = NULL;
    rt->batch.count_events = NULL;
    rt->batch.hit_counts = NULL;
    
    /* Build with minimal options */
    cl_int rc = clBuildProgram(rt->program, 1, &rt->device, NULL, NULL, NULL);
    
    if (rc != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(rt->program, rt->device, CL_PROGRAM_BUILD_LOG,
                              0, NULL, &log_size);
        char *log = malloc(log_size + 1);
        if (log) {
            clGetProgramBuildInfo(rt->program, rt->device, CL_PROGRAM_BUILD_LOG,
                                  log_size, log, NULL);
            log[log_size] = 0;
            cli_errmsg("GPU flattened kernel build failed:\n%s\n", log);
            free(log);
        }
        clReleaseProgram(rt->program);
        rt->program = NULL;
        return -1;
    }
    
    /* Create flattened kernel */
    rt->kernel = clCreateKernel(rt->program, "ac_scan_validate", &err);
    if (err != CL_SUCCESS || !rt->kernel) {
        cli_errmsg("GPU: Failed to create flattened kernel (%d)\n", err);
        clReleaseProgram(rt->program);
        rt->program = NULL;
        return -1;
    }

    

    rt->lsig_kernel = clCreateKernel(rt->program, "evaluate_logical_sigs", &err);
    if (err != CL_SUCCESS || !rt->lsig_kernel) {
        cli_dbgmsg("GPU: Logical signature kernel not available (%d)\n", err);
        rt->lsig_kernel = NULL;
        /* This is not fatal - continue without logical kernel */
    }
        
    rt->initialized = 1;
    cli_dbgmsg("GPU: Flattened kernel initialized successfully\n");
    return 0;
}


 

/* =========================
 * Scan entry point
 * ========================= */
/* ==========================================
 * OPTIMIZED KERNEL LAUNCH CONFIGURATION
//  * ========================================== */
 

 

/* ============================================
 * FLATTENED DFA BUILDER
 * Builds DFA with ALL signatures including metadata
 * ============================================ */
 struct gpu_flattened_dfa *gpu_build_flattened_dfa(struct cli_matcher *root)
{
     fprintf(stderr, "gpu_build_flattened_dfa ENTER: root=%p, type=%u\n", 
            (void*)root, root ? root->type : 999);
    cli_dbgmsg("GPU-DFA: Building flattened DFA for matcher %p\n", (void*)root);
    
    if (!root || !root->ac_root || !root->ac_nodetable || root->ac_nodes == 0) {
        cli_dbgmsg("GPU-DFA: AC not initialized\n");
        return NULL;
    }

    fprintf(stderr, "STEP 1: root=%p, ac_nodes=%u\n", (void*)root, root->ac_nodes);

    /* ============ CRITICAL: Reset state IDs ============ */
    for (uint32_t i = 0; i < root->ac_nodes; i++) {
        if (root->ac_nodetable[i])
            root->ac_nodetable[i]->gpu_state_id = UINT32_MAX;
    }
    fprintf(stderr, "STEP 2: Reset state IDs\n");

    /* Assign state IDs using existing function */
    struct cli_ac_node **states = NULL;
    uint32_t num_states = gpu_assign_state_ids(root->ac_root, &states);
    if (!states || num_states == 0) {
        cli_errmsg("GPU-DFA: state assignment failed\n");
        return NULL;
    }
    fprintf(stderr, "STEP 3: Assigned %u state IDs\n", num_states);

    if (num_states <= 1) {
        cli_errmsg("GPU-DFA: State assignment failed - only got %u states\n", num_states);
        free(states);
        return NULL;
    }
    
    /* Allocate flattened DFA structure */
    struct gpu_flattened_dfa *flat = cli_calloc(1, sizeof(*flat));
    if (!flat) {
        free(states);
        return NULL;
    } 
    fprintf(stderr, "STEP 4: Allocated flat structure\n");
    
    flat->states = num_states;
    flat->pat_count = root->gpu_patt_count;
    
    /* Allocate flattened DFA tables */
    flat->next = cli_malloc(num_states * 256 * sizeof(uint32_t));
    flat->out_index = cli_malloc(num_states * sizeof(uint32_t));
    flat->out_count = cli_calloc(num_states, sizeof(uint16_t));
    
    fprintf(stderr, "STEP 5: Allocated tables: next=%p, out_index=%p, out_count=%p\n", 
            flat->next, flat->out_index, flat->out_count);
    
    if (!flat->next || !flat->out_index || !flat->out_count) {
        cli_errmsg("GPU-DFA: Failed to allocate flattened DFA tables\n");
        goto error;
    }
    
    /* Count total outputs */
    uint32_t total_out = 0;
    for (uint32_t s = 0; s < num_states; s++) {
        struct cli_ac_node *node = states[s];
        struct cli_ac_list *l;
        
        for (l = node->list; l; l = l->next) total_out++;
        if (node->fail) {
            for (l = node->fail->list; l; l = l->next) total_out++;
        }
    }
    fprintf(stderr, "STEP 6: total_out=%u\n", total_out);

    if (total_out == 0) {
        cli_errmsg("GPU-DFA: No patterns found! Check if signatures are loaded.\n");
        goto error;
    }
 
    flat->out_pat = cli_malloc(total_out * sizeof(uint16_t));
    flat->sig_id = cli_calloc(total_out, sizeof(uint32_t));
    flat->part_no = cli_calloc(total_out, sizeof(uint32_t));
    fprintf(stderr, "STEP 7: Allocated output arrays\n");

    if (!flat->out_pat || !flat->sig_id || !flat->part_no) {
        cli_errmsg("GPU-DFA: Failed to allocate %u pattern tables\n", total_out);
        goto error;
    }
    
    /* Build transition table */
    memset(flat->next, 0, num_states * 256 * sizeof(uint32_t));
    fprintf(stderr, "STEP 8: Zeroed transition table\n");
    
    for (uint32_t state = 0; state < num_states; state++) {
        struct cli_ac_node *node = states[state];
        
        for (uint32_t c = 0; c < 256; c++) {
            if (node->trans && node->trans[c]) {
                flat->next[state * 256 + c] = node->trans[c]->gpu_state_id;
            } else if (node->fail) {
                flat->next[state * 256 + c] = 
                    flat->next[node->fail->gpu_state_id * 256 + c];
            }
        }
        // if (state % 10 == 0) fprintf(stderr, "  Processed state %u\n", state);
    }
    fprintf(stderr, "STEP 9: Built transition table\n");
    
    /* Fill pattern tables with metadata */
    uint32_t pat_idx = 0;
    uint32_t max_pattern_id = 0;
    for (uint32_t s = 0; s < num_states; s++) {
        struct cli_ac_node *node = states[s];
        struct cli_ac_list *l;
        for (l = node->list; l; l = l->next) {
            if (l->me->gpu_id > max_pattern_id)
                max_pattern_id = l->me->gpu_id;
        }
    }
    fprintf(stderr, "STEP 10: max_pattern_id=%u\n", max_pattern_id);

    uint8_t *seen_bitmap = NULL;
    if (max_pattern_id < 65536) {
        seen_bitmap = cli_calloc((max_pattern_id + 7) / 8, sizeof(uint8_t));
        fprintf(stderr, "STEP 11: Allocated bitmap\n");
    }

    for (uint32_t state = 0; state < num_states; state++) {
        struct cli_ac_node *node = states[state];
        struct cli_ac_list *l;
        
        flat->out_index[state] = pat_idx;
        
        for (l = node->list; l; l = l->next) {
            struct cli_ac_patt *patt = l->me;
            if (patt->gpu_id < flat->pat_count) {
                if (seen_bitmap) {
                    uint32_t byte = patt->gpu_id >> 3;
                    uint8_t bit = 1 << (patt->gpu_id & 7);
                    seen_bitmap[byte] |= bit;
                }
                flat->out_pat[pat_idx] = patt->gpu_id;
                flat->sig_id[pat_idx] = patt->sigid;
                flat->part_no[pat_idx] = patt->partno;
                flat->out_count[state]++;
                pat_idx++;
            }
        }
        
        if (node->fail) {
            for (l = node->fail->list; l; l = l->next) {
                struct cli_ac_patt *patt = l->me;
                
                if (patt->gpu_id >= flat->pat_count)
                    continue;
                    
                if (seen_bitmap) {
                    uint32_t byte = patt->gpu_id >> 3;
                    uint8_t bit = 1 << (patt->gpu_id & 7);
                    if (seen_bitmap[byte] & bit)
                        continue;
                    seen_bitmap[byte] |= bit;
                }
                
                flat->out_pat[pat_idx] = patt->gpu_id;
                flat->sig_id[pat_idx] = patt->sigid;
                flat->part_no[pat_idx] = patt->partno;
                flat->out_count[state]++;
                pat_idx++;
            }
        }
        flat->out_total = pat_idx;
        // if (state % 10 == 0) fprintf(stderr, "  Processed outputs for state %u, pat_idx=%u\n", state, pat_idx);
    }

    if (seen_bitmap) free(seen_bitmap);
    fprintf(stderr, "STEP 12: Completed pattern table filling, total_out=%u, pat_idx=%u\n", total_out, pat_idx);
    
    cli_dbgmsg("GPU-DFA: Flattened DFA built: %u states, %u patterns, %u outputs\n",
               flat->states, flat->pat_count, flat->out_total);
    
    free(states);
    fprintf(stderr, "STEP 13: Freed states\n");
    return flat;

error:
    if (flat->next) free(flat->next);
    if (flat->out_index) free(flat->out_index);
    if (flat->out_count) free(flat->out_count);
    if (flat->out_pat) free(flat->out_pat);
    if (flat->sig_id) free(flat->sig_id);
    if (flat->part_no) free(flat->part_no);
    free(flat);
    free(states);
    return NULL;
}

/* ============================================
 * FLATTENED DFA UPLOAD
 * Uploads DFA with signature metadata to GPU
 * ============================================ */
int gpu_rt_upload_flattened_dfa(struct gpu_rt *rt, struct gpu_flattened_dfa *dfa)
{
    cl_int err;
    uint32_t zero = 0;
    
    if (!rt || !dfa) return -1;
    if (rt->dfa_uploaded) {
        cli_dbgmsg("GPU-Flatten: Already uploaded\n");
        return 0;
    }
    
    cli_dbgmsg("GPU-Flatten: Uploading %u states, %u patterns, %u outputs\n",
               dfa->states, dfa->pat_count, dfa->out_total);


 
    
    /* Calculate buffer sizes */
    size_t next_size = dfa->states * 256 * sizeof(uint32_t);
    size_t index_size = dfa->states * sizeof(uint32_t);
    size_t count_size = dfa->states * sizeof(uint16_t);
    if (dfa->out_total == 0) {
    cli_errmsg("GPU-Flatten: out_total is 0, cannot upload\n");
    return -1;
}
    size_t pat_size = dfa->out_total * sizeof(uint16_t);
    size_t sig_size = dfa->out_total * sizeof(uint32_t);
    size_t part_size = dfa->out_total * sizeof(uint32_t);

//     fprintf(stderr, "GPU-Flatten: next_size=%zu index_size=%zu count_size=%zu pat_size=%zu\n",
//         next_size, index_size, count_size, pat_size);
// fprintf(stderr, "GPU-Flatten: states=%u pat_count=%u out_total=%u\n",
//         dfa->states, dfa->pat_count, dfa->out_total);
// fprintf(stderr, "GPU-Flatten: scan_buffer=%zu\n", (size_t)64 * 1024 * 1024);
    

    /* Batch create all buffers first */
    rt->dfa_next = clCreateBuffer(rt->context, CL_MEM_READ_ONLY,
                                  next_size, NULL, &err);
    if (err != CL_SUCCESS) goto error;
    
    rt->dfa_out_index = clCreateBuffer(rt->context, CL_MEM_READ_ONLY,
                                       index_size, NULL, &err);
    if (err != CL_SUCCESS) goto error;
    
    rt->dfa_out_count = clCreateBuffer(rt->context, CL_MEM_READ_ONLY,
                                       count_size, NULL, &err);
    if (err != CL_SUCCESS) goto error;
    
    rt->dfa_out_pat = clCreateBuffer(rt->context, CL_MEM_READ_ONLY,
                                     pat_size, NULL, &err);
    if (err != CL_SUCCESS) goto error;
    
    rt->dfa_sig_id = clCreateBuffer(rt->context, CL_MEM_READ_ONLY,
                                    sig_size, NULL, &err);
    if (err != CL_SUCCESS) goto error;
    
    rt->dfa_part_no = clCreateBuffer(rt->context, CL_MEM_READ_ONLY,
                                     part_size, NULL, &err);
    if (err != CL_SUCCESS) goto error;
    
    /* Now queue all writes asynchronously */
    clEnqueueWriteBuffer(rt->queue, rt->dfa_next, CL_FALSE, 0,
                         next_size, dfa->next, 0, NULL, NULL);
    clEnqueueWriteBuffer(rt->queue, rt->dfa_out_index, CL_FALSE, 0,
                         index_size, dfa->out_index, 0, NULL, NULL);
    clEnqueueWriteBuffer(rt->queue, rt->dfa_out_count, CL_FALSE, 0,
                         count_size, dfa->out_count, 0, NULL, NULL);
    clEnqueueWriteBuffer(rt->queue, rt->dfa_out_pat, CL_FALSE, 0,
                         pat_size, dfa->out_pat, 0, NULL, NULL);
    clEnqueueWriteBuffer(rt->queue, rt->dfa_sig_id, CL_FALSE, 0,
                         sig_size, dfa->sig_id, 0, NULL, NULL);
    clEnqueueWriteBuffer(rt->queue, rt->dfa_part_no, CL_FALSE, 0,
                         part_size, dfa->part_no, 0, NULL, NULL);
    
    /* Hit buffers */
     
      
    if (err != CL_SUCCESS) goto error;
    
 
    
   
    
    /* Store metadata */
    rt->dfa_states = dfa->states;
    rt->patt_count = dfa->out_total;
    rt->dfa_uploaded = 1;
    rt->out_total = dfa->out_total;
    cli_dbgmsg("GPU-Flatten: Upload successful!\n");
    return 0;
    
error:
    cli_errmsg("GPU-Flatten: Upload failed (err=%d)\n", err);
    /* Only release what THIS call created, not the entire runtime */
    if (rt->dfa_next) { clReleaseMemObject(rt->dfa_next); rt->dfa_next = NULL; }
    if (rt->dfa_out_index) { clReleaseMemObject(rt->dfa_out_index); rt->dfa_out_index = NULL; }
    if (rt->dfa_out_count) { clReleaseMemObject(rt->dfa_out_count); rt->dfa_out_count = NULL; }
    if (rt->dfa_out_pat) { clReleaseMemObject(rt->dfa_out_pat); rt->dfa_out_pat = NULL; }
    if (rt->dfa_sig_id) { clReleaseMemObject(rt->dfa_sig_id); rt->dfa_sig_id = NULL; }
    if (rt->dfa_part_no) { clReleaseMemObject(rt->dfa_part_no); rt->dfa_part_no = NULL; }
    if (rt->scan_buffer) { clReleaseMemObject(rt->scan_buffer); rt->scan_buffer = NULL; }
    rt->dfa_uploaded = 0;
    return -1;
}



void gpu_rt_destroy(struct gpu_rt *rt)
{
    if (!rt) return;

    /* DFA buffers */
    if (rt->dfa_next)        clReleaseMemObject(rt->dfa_next);
    if (rt->dfa_out_index)   clReleaseMemObject(rt->dfa_out_index);
    if (rt->dfa_out_count)   clReleaseMemObject(rt->dfa_out_count);
    if (rt->dfa_out_pat)     clReleaseMemObject(rt->dfa_out_pat);

    for (uint32_t i = 0; i < rt->batch.count; i++) {
    if (rt->batch.buffers[i]) {
        free((void*)rt->batch.buffers[i]);
        rt->batch.buffers[i] = NULL;  // MUST null after free
    }
}
 
    
    if (rt->batch.file_offsets) free(rt->batch.file_offsets);
    
    /* Working buffers */   
    if (rt->is_simple)       clReleaseMemObject(rt->is_simple);
    if (rt->scan_buffer)     clReleaseMemObject(rt->scan_buffer);
    
    /* Signature metadata buffers */
    if (rt->dfa_sig_id)      clReleaseMemObject(rt->dfa_sig_id);
    if (rt->dfa_part_no)     clReleaseMemObject(rt->dfa_part_no);
    
    /* Batch buffers */
    if (rt->scan_buffer) clReleaseMemObject(rt->scan_buffer);
 
    if (rt->batch.lengths) free(rt->batch.lengths); 
    if (rt->batch.kernel_events) free(rt->batch.kernel_events);
    if (rt->batch.count_events) free(rt->batch.count_events);
    if (rt->batch.hit_counts) free(rt->batch.hit_counts);
 
 
    /* Async buffers */
    if (rt->async_buf_is_temp && rt->async_buf) 
        clReleaseMemObject(rt->async_buf);
    if (rt->async_kernel_event) clReleaseEvent(rt->async_kernel_event);
    if (rt->async_count_event) clReleaseEvent(rt->async_count_event);
    
    /* OpenCL objects */
    if (rt->kernel)          clReleaseKernel(rt->kernel);
    if (rt->program)         clReleaseProgram(rt->program);
    if (rt->queue)           clReleaseCommandQueue(rt->queue);
    if (rt->context)         clReleaseContext(rt->context);
    
    memset(rt, 0, sizeof(*rt));
}
 
