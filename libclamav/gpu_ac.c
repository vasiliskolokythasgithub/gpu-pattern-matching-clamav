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




 

/**
 * Upload logical signature metadata to GPU
 * Call this after gpu_upload_pattern_metadata
 */

 int gpu_upload_logical_signatures(struct gpu_rt *rt, struct cli_matcher *root, uint32_t matcher_idx)
{
    struct gpu_matcher_data *m = &rt->matchers[matcher_idx];
    cl_int err;

    if (!rt || !root) {
        fprintf(stderr, "  GPU LSIG: rt or root is NULL\n");
        return -1;
    }
    
    fprintf(stderr, "  GPU LSIG: root->ac_lsigtable=%p, ac_lsigs=%u (matcher %u)\n", 
            (void*)root->ac_lsigtable, root->ac_lsigs, matcher_idx);
    fflush(stderr);
    
    if (!rt || !root || !root->ac_lsigtable) {
        fprintf(stderr, "  GPU LSIG: No logical signatures (null)\n");
        m->num_lsigs = 0;
        return 0;
    }
    
    fprintf(stderr, "  GPU LSIG: Counting logical signatures...\n");
    fflush(stderr);
    
    /* Count logical signatures */
    uint32_t num_lsigs = 0;
    while (num_lsigs < root->ac_lsigs && 
           root->ac_lsigtable[num_lsigs] != NULL) {
        num_lsigs++;
        if (num_lsigs % 1000 == 0) {
            // fprintf(stderr, "    Counted %u logical signatures\n", num_lsigs);
            fflush(stderr);
        }
    }
    
    fprintf(stderr, "  GPU LSIG: Found %u logical signatures\n", num_lsigs);
    fflush(stderr);

    fprintf(stderr, "=== GPU LSIG UPLOAD (matcher %u) ===\n", matcher_idx);
    fprintf(stderr, "root=%p\n", (void*)root);
    
    if (!rt || !root) {
        fprintf(stderr, "GPU LSIG: rt or root is NULL\n");
        return -1;
    }
    
    fprintf(stderr, "root->ac_lsigtable=%p\n", (void*)root->ac_lsigtable);
    fprintf(stderr, "root->ac_lsigs=%u\n", root->ac_lsigs);
    
    if (!rt || !root || !root->ac_lsigtable) {
        fprintf(stderr, "GPU LSIG: No logical signatures (null)\n");
        m->num_lsigs = 0;
        return 0;
    }
    
    /* Count logical signatures */ 
    num_lsigs = 0;
    while (num_lsigs < root->ac_lsigs && 
           root->ac_lsigtable[num_lsigs] != NULL) {
        num_lsigs++;
    }
    
    fprintf(stderr, "GPU LSIG: Found %u logical signatures out of %u\n", 
            num_lsigs, root->ac_lsigs);
    
    if (num_lsigs == 0) {
        m->num_lsigs = 0;
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
        if (lsig->tdb.subsigs == 1 && 
            !lsig->tdb.container && 
            !lsig->tdb.filesize && 
            !lsig->tdb.ep &&
            !lsig->tdb.nos) {
            meta->expr_length = 0;
            continue;
        }
        
        meta->sig_id = i + 1;  
        meta->num_subsigs = lsig->tdb.subsigs;

        /* Debug for problematic signatures */
        if (lsig->virname && strstr(lsig->virname, "Mediaget")) {
            fprintf(stderr, "FIXED: LSIG[%u] %s: tdb.subsigs=%u, actual_subsigs=%u\n",
                    i, lsig->virname, lsig->tdb.subsigs, meta->num_subsigs);
        }
        meta->expr_offset = current_pos;
        
        if (lsig->virname && strstr(lsig->virname, "ZxShell-10")) {
            fprintf(stderr, "ZxShell LSIG[%u]: sig_id=%u, subsigs=%u, expr='%s'\n",
                    i, i+1, lsig->tdb.subsigs, lsig->u.logic ? lsig->u.logic : "NULL");
                    fprintf(stderr, "  nos=[%u,%u], handlertype=%u\n",
        lsig->tdb.nos ? lsig->tdb.nos[0] : 0,
        lsig->tdb.nos ? lsig->tdb.nos[1] : 0,
        lsig->tdb.handlertype ? lsig->tdb.handlertype[0] : 0);
            fprintf(stderr, "  container=%u, filesize=[%u,%u], ep=[%u,%u], nos=[%u,%u]\n",
                    lsig->tdb.container ? lsig->tdb.container[0] : 0,
                    lsig->tdb.filesize ? lsig->tdb.filesize[0] : 0,
                    lsig->tdb.filesize ? lsig->tdb.filesize[1] : 0,
                    lsig->tdb.ep ? lsig->tdb.ep[0] : 0,
                    lsig->tdb.ep ? lsig->tdb.ep[1] : 0,
                    lsig->tdb.nos ? lsig->tdb.nos[0] : 0,
                    lsig->tdb.nos ? lsig->tdb.nos[1] : 0);
            // Print subsig patterns
            for (uint32_t s = 0; s < root->ac_patterns; s++) {
                struct cli_ac_patt *p = root->ac_pattable[s];
                if (p && p->lsigid[0] && p->lsigid[1] == i+1) {
                    fprintf(stderr, "  subsig[%u]: len=%u, depth=%u, pattern: ",
                            p->lsigid[2], p->length[0], p->depth);
                    for (int j = 0; j < p->length[0] && j < 8; j++)
                        fprintf(stderr, "%04x ", p->pattern[j]);
                    fprintf(stderr, "\n");
                }
            }
        }

        /* Handle virus name */
        if (lsig->virname) {
            meta->virname_offset = m->virname_pool_size;
            meta->virname_len = strlen(lsig->virname);
            
            char *new_pool = realloc(m->h_virname_pool, m->virname_pool_size + meta->virname_len + 1);
            if (!new_pool) {
                free(h_metas);
                free(h_bytecode);
                return -1;
            }
            m->h_virname_pool = new_pool;
            
            strcpy(m->h_virname_pool + m->virname_pool_size, lsig->virname);
            m->virname_pool_size += meta->virname_len + 1;
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
    m->d_lsig_metas = clCreateBuffer(rt->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       num_lsigs * sizeof(gpu_lsig_meta_t),
                                       h_metas, &err);
    if (err != CL_SUCCESS) goto error;
    
    m->d_expr_bytecode = clCreateBuffer(rt->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          current_pos * sizeof(gpu_expr_inst_t),
                                          h_bytecode, &err);
    if (err != CL_SUCCESS) goto error;
    
    m->num_lsigs = num_lsigs;
    m->expr_bytecode_size = current_pos;
    
    fprintf(stderr, "GPU LSIG: Uploaded %u signatures, %u expr nodes for matcher %u\n",
            num_lsigs, current_pos, matcher_idx);
    
    fprintf(stderr, "=== LOGICAL SIGNATURES IN ROOT (matcher %u) ===\n", matcher_idx);
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
    cli_errmsg("GPU: Failed to upload logical signatures for matcher %u: %d\n", matcher_idx, err);
    free(h_metas);
    free(h_bytecode);
    return -1;
}


int gpu_upload_pattern_metadata(struct gpu_rt *rt, struct cli_matcher *root, uint32_t matcher_idx)
{
    struct gpu_matcher_data *m = &rt->matchers[matcher_idx];
    
    fprintf(stderr, "=== gpu_upload_pattern_metadata START (matcher %u) ===\n", matcher_idx);
    fprintf(stderr, "rt=%p, root=%p\n", (void*)rt, (void*)root);
    fprintf(stderr, "root->gpu_patt_count=%u\n", root->gpu_patt_count);
    fprintf(stderr, "root->ac_patterns=%u\n", root->ac_patterns);
    
    cl_int err;
    uint32_t np = root->gpu_patt_count;
    
    if (np == 0) {
        fprintf(stderr, "gpu_upload_pattern_metadata: ERROR - gpu_patt_count is 0\n");
        return -1;
    }
    
    /* Pass 1: Calculate total sizes and debug logical patterns */
    uint32_t total_pat_bytes = 0;
    uint32_t total_pfx_bytes = 0;
    uint32_t total_vn_bytes = 0;
    uint32_t logical_pattern_count = 0;
    
    fprintf(stderr, "  Calculating sizes for %u patterns...\n", np);
    
    /* Build a mapping from logical signature ID to subsig count */
    uint32_t *lsig_subsig_counts = NULL;
    if (root->ac_lsigs > 0 && root->ac_lsigtable) {
        lsig_subsig_counts = calloc(root->ac_lsigs, sizeof(uint32_t));
        if (lsig_subsig_counts) {
            for (uint32_t i = 0; i < root->ac_lsigs; i++) {
                if (root->ac_lsigtable[i]) {
                    lsig_subsig_counts[i] = root->ac_lsigtable[i]->tdb.subsigs;
                    //fprintf(stderr, "  LSIG[%u]: has %u subsignatures\n", i, lsig_subsig_counts[i]);
                }
            }
        }
    }
    
    for (uint32_t i = 0; i < np; i++) {
        struct cli_ac_patt *p = root->gpu_patt_lookup[i];
        if (!p) {
            fprintf(stderr, "  WARNING: gpu_patt_lookup[%u] is NULL\n", i);
            continue;
        }
        if (i != p->gpu_id) {
        fprintf(stderr, "WARNING: Pattern index mismatch: i=%u, gpu_id=%u\n", i, p->gpu_id);
        }
            
        
        if (p->lsigid[0] && p->lsigid[1] > root->ac_lsigs) {
            fprintf(stderr, "ERROR: pattern %u has lsigid[1]=%u > num_lsigs=%u\n",
                    i, p->lsigid[1], root->ac_lsigs);
            
        }
        
        /* Check pattern bytes pointer */
        if (!p->pattern && p->length[0] > 0) {
            fprintf(stderr, "ERROR: pattern %u has NULL pattern bytes (len=%u)\n", 
                    i, p->length[0]);
           
        }
        total_pat_bytes += p->length[0];
        total_pfx_bytes += p->prefix_length[0];
        if (p->virname)
            total_vn_bytes += strlen(p->virname) + 1;
        
        /* Debug logical patterns and fix partno/parts if needed */
        if (p->lsigid[0] > 0) {
            logical_pattern_count++;
            uint32_t lsig_idx = p->lsigid[1] - 1;  /* Convert to 0-based */
            uint32_t subsig = p->lsigid[2];
            

                if (p->lsigid[1] == 0) {
        fprintf(stderr, "  WARNING: lsigid[1]=0 for pattern %u - skipping fix\n", i);
        continue;  // or just skip the fix block
    }

            /* CRITICAL: If partno/parts are 0, try to fix them from lsig table */
            if ((p->partno == 0 || p->parts == 0) && lsig_subsig_counts && lsig_idx < root->ac_lsigs) {
                uint32_t total_subsigs = lsig_subsig_counts[lsig_idx];
                if (total_subsigs > 0) {
                    p->partno = subsig + 1;  /* 1-based part number */
                    p->parts = total_subsigs;
                }
            }
            
            /* Also warn if still zero */
            if (p->partno == 0 || p->parts == 0) {
                fprintf(stderr, "  WARNING: Logical pattern has partno=%u, parts=%u - this will break GPU detection!\n",
                        p->partno, p->parts);
            }
        }
        
        if (i < 5) {
            fprintf(stderr, "    pattern[%u]: len=%u, pfx_len=%u, virname=%s, lsigid[0]=%u\n", 
                    i, p->length[0], p->prefix_length[0], 
                    p->virname ? p->virname : "NULL", p->lsigid[0]);
        }
    }
    
    fprintf(stderr, "  total_pat_bytes=%u, total_pfx_bytes=%u, total_vn_bytes=%u\n",
            total_pat_bytes, total_pfx_bytes, total_vn_bytes);
    fprintf(stderr, "  logical_pattern_count=%u\n", logical_pattern_count);
    
    /* Ensure minimums for buffer creation */
    if (total_pat_bytes == 0) total_pat_bytes = 1;
    if (total_pfx_bytes == 0) total_pfx_bytes = 1;
    if (total_vn_bytes == 0) total_vn_bytes = 1;
    
    /* Allocate host arrays */
    fprintf(stderr, "  Allocating host arrays...\n");
    
    gpu_pattern_t *h_patterns = calloc(np, sizeof(gpu_pattern_t));
    uint16_t *h_pat_bytes = malloc(total_pat_bytes * sizeof(uint16_t));
    uint16_t *h_pfx_bytes = malloc(total_pfx_bytes * sizeof(uint16_t));
    char *h_vn_pool = malloc(total_vn_bytes);
    
    if (!h_patterns || !h_pat_bytes || !h_pfx_bytes || !h_vn_pool) {
        fprintf(stderr, "  ERROR: Failed to allocate host arrays\n");
        goto fail_free_arrays;
    }
    
    /* Pass 2: Fill arrays */
    fprintf(stderr, "  Filling host arrays...\n");
    
    uint32_t pat_off = 0, pfx_off = 0, vn_off = 0;
    uint32_t max_depth = root->ac_maxdepth;
    
    for (uint32_t i = 0; i < np; i++) {
        struct cli_ac_patt *p = root->gpu_patt_lookup[i];
        if (!p) continue;
        
        gpu_pattern_t *gp = &h_patterns[i];
        
        /* Copy all pattern fields */
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
        gp->lsigid[2] = p->lsigid[2];
        gp->boundary = p->boundary;
        gp->sigopts = p->sigopts;
        gp->mindist = p->mindist;
        gp->maxdist = p->maxdist;
        gp->has_regex = 0;
        gp->special_pattern = p->special_pattern;
        
        /* Depth calculation */
        if (p->lsigid[0] > 0 && p->depth > 0) {
            gp->depth = p->depth;
        } else {
            gp->depth = (p->length[0] < max_depth) ? p->length[0] : max_depth;
        }
        
        /* Debug fixed logical patterns */
        if (p->lsigid[0] > 0) {
            /* Get logical signature metadata for regex flag */
            uint32_t lsig_idx = p->lsigid[1] - 1;
            if (lsig_idx < root->ac_lsigs && root->ac_lsigtable[lsig_idx]) {
                gp->has_regex = root->ac_lsigtable[lsig_idx]->has_regex;
            }
        }
        
        /* Map offset types */
        if (p->offdata[0] == CLI_OFF_ANY || p->offdata[0] == 0) {
            gp->offdata0 = GPU_OFF_ANY;
        } else if (p->offdata[0] == CLI_OFF_NONE) {
            gp->offdata0 = GPU_OFF_NONE;
        } else if (p->offdata[0] == CLI_OFF_ABSOLUTE) {
            gp->offdata0 = GPU_OFF_ABSOLUTE;
        } else {
            gp->offdata0 = GPU_OFF_ANY;
        }
        
        /* Copy character metadata */
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
                    fprintf(stderr, "  ERROR: pat_off overflow: %u >= %u\n", pat_off, total_pat_bytes);
                    goto fail_free_arrays;
                }
                h_pat_bytes[pat_off++] = p->pattern[j];
            }
        }
        
        /* Prefix bytes */
        gp->prefix_offset = pfx_off;
        if (p->prefix) {
            for (uint16_t j = 0; j < p->prefix_length[0]; j++) {
                if (pfx_off >= total_pfx_bytes) {
                    fprintf(stderr, "  ERROR: pfx_off overflow: %u >= %u\n", pfx_off, total_pfx_bytes);
                    goto fail_free_arrays;
                }
                h_pfx_bytes[pfx_off++] = p->prefix[j];
            }
        }
        
        /* Virname */
        if (p->virname) {
            gp->virname_offset = vn_off;
            gp->virname_len = strlen(p->virname);
            
            if (vn_off + gp->virname_len + 1 > total_vn_bytes) {
                fprintf(stderr, "  ERROR: vn_off overflow: %u + %u > %u\n",
                        vn_off, gp->virname_len + 1, total_vn_bytes);
                goto fail_free_arrays;
            }
            
            memcpy(h_vn_pool + vn_off, p->virname, gp->virname_len + 1);
            vn_off += gp->virname_len + 1;
        } else {
            gp->virname_offset = 0;
            gp->virname_len = 0;
        }
    }
    
    fprintf(stderr, "  Final offsets: pat_off=%u/%u, pfx_off=%u/%u, vn_off=%u/%u\n",
            pat_off, total_pat_bytes, pfx_off, total_pfx_bytes, vn_off, total_vn_bytes);
    
    /* Store sizes in matcher struct */
    m->num_pattern_bytes = pat_off;
    m->num_prefix_bytes = pfx_off;
    m->num_patterns = np;
    
    /* Create GPU buffers */
    fprintf(stderr, "  Creating GPU buffers...\n");
    
    m->d_patterns = clCreateBuffer(rt->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     np * sizeof(gpu_pattern_t), h_patterns, &err);
    if (err != CL_SUCCESS) goto fail_free_arrays;
    fprintf(stderr, "    d_patterns: created\n");
    
    m->d_pattern_bytes = clCreateBuffer(rt->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          pat_off * sizeof(uint16_t), h_pat_bytes, &err);
    if (err != CL_SUCCESS) goto fail_release_patterns;
    fprintf(stderr, "    d_pattern_bytes: created\n");
    
    m->d_prefix_bytes = clCreateBuffer(rt->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         pfx_off * sizeof(uint16_t), h_pfx_bytes, &err);
    if (err != CL_SUCCESS) goto fail_release_pattern_bytes;
    fprintf(stderr, "    d_prefix_bytes: created\n");
    
    /* Create virname pool */
    fprintf(stderr, "  Creating virname pool of size %u\n", vn_off);
    m->h_virname_pool = malloc(vn_off);
    if (!m->h_virname_pool) {
        fprintf(stderr, "  ERROR: Failed to allocate host virname pool\n");
        goto fail_release_prefix_bytes;
    }
    memcpy(m->h_virname_pool, h_vn_pool, vn_off);
    m->virname_pool_size = vn_off;
    
    m->d_virname_pool = clCreateBuffer(rt->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         m->virname_pool_size, m->h_virname_pool, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "    d_virname_pool: err=%d\n", err);
        goto fail_free_host_pool;
    }
    fprintf(stderr, "    d_virname_pool: created\n");
    
    /* Cleanup */
    free(h_patterns);
    free(h_pat_bytes);
    free(h_pfx_bytes);
    free(h_vn_pool);
    if (lsig_subsig_counts) free(lsig_subsig_counts);
    
    fprintf(stderr, "gpu_upload_pattern_metadata: SUCCESS - %u patterns uploaded for matcher %u\n", np, matcher_idx);
    return 0;
    
/* Error handling */
fail_release_prefix_bytes:
    clReleaseMemObject(m->d_prefix_bytes);
    m->d_prefix_bytes = NULL;
fail_release_pattern_bytes:
    clReleaseMemObject(m->d_pattern_bytes);
    m->d_pattern_bytes = NULL;
fail_release_patterns:
    clReleaseMemObject(m->d_patterns);
    m->d_patterns = NULL;
fail_free_host_pool:
    free(m->h_virname_pool);
    m->h_virname_pool = NULL;
fail_free_arrays:
    if (h_patterns) free(h_patterns);
    if (h_pat_bytes) free(h_pat_bytes);
    if (h_pfx_bytes) free(h_pfx_bytes);
    if (h_vn_pool) free(h_vn_pool);
    if (lsig_subsig_counts) free(lsig_subsig_counts);
    fprintf(stderr, "gpu_upload_pattern_metadata: FAILED for matcher %u\n", matcher_idx);
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
    
    /* Get the selected matcher data */
    //CHANGE THIS
    struct gpu_matcher_data *m = &rt->matchers[0];

    if (!rt || !m->uploaded) {
        
        return GPU_RESULT_BREAK;
    }
        
    /* Get TDB information from context */
    if (tinfo && tinfo->status == 1) {
        entry_point = tinfo->exeinfo.ep;
        is_pe_target = 1;
    }
    
    if (ctx && ctx->recursion_level >= 0) {
        container_type = cli_recursion_stack_get_type(ctx, -1);
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
    fflush(stdout);

    /* Calculate launch params */
    uint32_t chunk_size;

    if (file_length >= 128 * 1024 * 1024)      chunk_size = 524288;
    else if (file_length >= 64 * 1024 * 1024)   chunk_size = 262144;
    else if (file_length >= 16 * 1024 * 1024)   chunk_size = 131072;
    else if (file_length >= 4 * 1024 * 1024)    chunk_size = 65536;
    else if (file_length >= 1 * 1024 * 1024)    chunk_size = 32768;
    else                                         chunk_size = 16384;
    if (maxpatlen < 2) maxpatlen = 2;

    /* Ensure chunk_size is at least maxpatlen to avoid underflow */
    // if (chunk_size < maxpatlen) {
    //     chunk_size = maxpatlen;
    // }

    if (chunk_size < maxpatlen * 2) {
    chunk_size = maxpatlen * 2;
    }
    uint32_t file_offset = 0;
    uint32_t stride = chunk_size - (maxpatlen - 1);
    uint32_t chunks = (file_length + stride - 1) / stride;
    size_t gws = chunks;
    if (gws == 0) gws = 1;
    gws = ((gws + 255) / 256) * 256;
    uint32_t max_trackers = GPU_MAX_TRACKERS;

    uint32_t is_elf_target = 0;
    if (file_length >= 4 && file_buffer[0] == 0x7F && 
        file_buffer[1] == 'E' && file_buffer[2] == 'L' && file_buffer[3] == 'F') {
        is_elf_target = 1;
    }

    /* Set kernel args for MAIN kernel */
    int a = 0; 

    err = clSetKernelArg(rt->kernel, a++, sizeof(cl_mem), &rt->scan_buffer);
    if (err != CL_SUCCESS) fprintf(stderr, "ERROR: Failed to set scan_buffer arg\n");

    err = clSetKernelArg(rt->kernel, a++, sizeof(uint32_t), &file_offset);
    if (err != CL_SUCCESS) fprintf(stderr, "ERROR: Failed to set file_offset arg\n");

    err = clSetKernelArg(rt->kernel, a++, sizeof(uint32_t), &file_length);
    if (err != CL_SUCCESS) fprintf(stderr, "ERROR: Failed to set file_length arg\n");

    /* Use the matcher's DFA buffers */
    err = clSetKernelArg(rt->kernel, a++, sizeof(cl_mem), &m->dfa_next);
    if (err != CL_SUCCESS) fprintf(stderr, "ERROR: Failed to set dfa_next arg\n");

    err = clSetKernelArg(rt->kernel, a++, sizeof(cl_mem), &m->dfa_out_index);
    if (err != CL_SUCCESS) fprintf(stderr, "ERROR: Failed to set dfa_out_index arg\n");

    err = clSetKernelArg(rt->kernel, a++, sizeof(cl_mem), &m->dfa_out_count);
    if (err != CL_SUCCESS) fprintf(stderr, "ERROR: Failed to set dfa_out_count arg\n");

    err = clSetKernelArg(rt->kernel, a++, sizeof(cl_mem), &m->dfa_out_pat);
    if (err != CL_SUCCESS) fprintf(stderr, "ERROR: Failed to set dfa_out_pat arg\n");

    err = clSetKernelArg(rt->kernel, a++, sizeof(uint32_t), &m->dfa_states);
    if (err != CL_SUCCESS) fprintf(stderr, "ERROR: Failed to set dfa_states arg\n");

    err = clSetKernelArg(rt->kernel, a++, sizeof(cl_mem), &m->d_patterns);
    if (err != CL_SUCCESS) fprintf(stderr, "ERROR: Failed to set d_patterns arg\n");

    err = clSetKernelArg(rt->kernel, a++, sizeof(uint32_t), &m->num_patterns);
    if (err != CL_SUCCESS) fprintf(stderr, "ERROR: Failed to set num_patterns arg\n");

    err = clSetKernelArg(rt->kernel, a++, sizeof(cl_mem), &m->d_pattern_bytes);
    if (err != CL_SUCCESS) fprintf(stderr, "ERROR: Failed to set d_pattern_bytes arg\n");

    err = clSetKernelArg(rt->kernel, a++, sizeof(cl_mem), &m->d_prefix_bytes);
    if (err != CL_SUCCESS) fprintf(stderr, "ERROR: Failed to set d_prefix_bytes arg\n");

    err = clSetKernelArg(rt->kernel, a++, sizeof(cl_mem), &rt->d_tracker_pool);
    if (err != CL_SUCCESS) fprintf(stderr, "ERROR: Failed to set d_tracker_pool arg\n");

    err = clSetKernelArg(rt->kernel, a++, sizeof(cl_mem), &rt->d_tracker_count);
    if (err != CL_SUCCESS) fprintf(stderr, "ERROR: Failed to set d_tracker_count arg\n");

    err = clSetKernelArg(rt->kernel, a++, sizeof(uint32_t), &max_trackers);
    if (err != CL_SUCCESS) fprintf(stderr, "ERROR: Failed to set max_trackers arg\n");

    err = clSetKernelArg(rt->kernel, a++, sizeof(uint32_t), &chunk_size);
    if (err != CL_SUCCESS) fprintf(stderr, "ERROR: Failed to set chunk_size arg\n");

    err = clSetKernelArg(rt->kernel, a++, sizeof(uint32_t), &stride);
    if (err != CL_SUCCESS) fprintf(stderr, "ERROR: Failed to set stride arg\n");

    err = clSetKernelArg(rt->kernel, a++, sizeof(cl_mem), &rt->d_result);
    if (err != CL_SUCCESS) fprintf(stderr, "ERROR: Failed to set d_result arg\n");

    err = clSetKernelArg(rt->kernel, a++, sizeof(uint32_t), &m->num_pattern_bytes);
    if (err != CL_SUCCESS) fprintf(stderr, "ERROR: Failed to set num_pattern_bytes arg\n");

    err = clSetKernelArg(rt->kernel, a++, sizeof(uint32_t), &m->num_prefix_bytes);
    if (err != CL_SUCCESS) fprintf(stderr, "ERROR: Failed to set num_prefix_bytes arg\n");

    err = clSetKernelArg(rt->kernel, a++, sizeof(uint32_t), &m->out_total);
    if (err != CL_SUCCESS) fprintf(stderr, "ERROR: Failed to set out_total arg\n");

    err = clSetKernelArg(rt->kernel, a++, sizeof(cl_mem), &m->d_lsig_metas);
    err = clSetKernelArg(rt->kernel, a++, sizeof(uint32_t), &m->num_lsigs);
    err = clSetKernelArg(rt->kernel, a++, sizeof(uint32_t), &container_type);
    err = clSetKernelArg(rt->kernel, a++, sizeof(uint32_t), &entry_point);
    err = clSetKernelArg(rt->kernel, a++, sizeof(uint32_t), &is_pe_target);
    err = clSetKernelArg(rt->kernel, a++, sizeof(uint32_t), &is_elf_target);

    fflush(stderr);
 
    fprintf(stderr, "dfa_states = %u\n", m->dfa_states);
    fprintf(stderr, "out_total = %u\n", m->out_total);
    fprintf(stderr, "num_patterns = %u\n", m->num_patterns);
    fprintf(stderr, "num_pattern_bytes = %u\n", m->num_pattern_bytes);
    fprintf(stderr, "num_prefix_bytes = %u\n", m->num_prefix_bytes);
    fprintf(stderr, "num_lsigs = %u\n", m->num_lsigs);
    fprintf(stderr, "expr_bytecode_size = %u\n", m->expr_bytecode_size);
    fprintf(stderr, "chunk_size = %u\n", chunk_size);
    fprintf(stderr, "stride = %u\n", stride);
    fprintf(stderr, "max_trackers = %u\n", max_trackers);


    fprintf(stderr, "Buffer check:\n");
fprintf(stderr, "  scan_buffer: %p\n", (void*)rt->scan_buffer);
fprintf(stderr, "  dfa_next: %p\n", (void*)m->dfa_next);
fprintf(stderr, "  dfa_out_index: %p\n", (void*)m->dfa_out_index);
fprintf(stderr, "  dfa_out_count: %p\n", (void*)m->dfa_out_count);
fprintf(stderr, "  dfa_out_pat: %p\n", (void*)m->dfa_out_pat);
fprintf(stderr, "  d_patterns: %p\n", (void*)m->d_patterns);
fprintf(stderr, "  d_pattern_bytes: %p\n", (void*)m->d_pattern_bytes);
fprintf(stderr, "  d_prefix_bytes: %p\n", (void*)m->d_prefix_bytes);
fprintf(stderr, "  d_tracker_pool: %p\n", (void*)rt->d_tracker_pool);
fprintf(stderr, "  d_tracker_count: %p\n", (void*)rt->d_tracker_count);
fprintf(stderr, "  d_result: %p\n", (void*)rt->d_result);
fprintf(stderr, "  d_lsig_metas: %p\n", (void*)m->d_lsig_metas);

    size_t actual_next_size;
    clGetMemObjectInfo(m->dfa_next, CL_MEM_SIZE, sizeof(actual_next_size), &actual_next_size, NULL);
    fprintf(stderr, "PRE-LAUNCH CHECK: dfa_states=%u, buffer_size=%zu, expected_size=%zu\n",
            m->dfa_states, actual_next_size, (size_t)m->dfa_states * 256 * sizeof(uint32_t));
    
    /* Launch MAIN kernel */
    size_t local_work_size = 64;
    err = clEnqueueNDRangeKernel(rt->queue, rt->kernel, 1, NULL,
                                  &gws, &local_work_size, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "GPU: Main kernel launch failed: %d\n", err);
        return GPU_RESULT_BREAK;
    }

    /* Force completion and flush */
    clFinish(rt->queue);
    fflush(stdout);

    /* If we have logical signatures, run evaluation kernel */
    if (m->num_lsigs > 0 && rt->lsig_kernel) {
        /* Get tracker count from main kernel */
        uint32_t tracker_count;
        clEnqueueReadBuffer(rt->queue, rt->d_tracker_count, CL_TRUE,
                            0, sizeof(uint32_t), &tracker_count, 0, NULL, NULL);
        fprintf(stderr, "GPU: tracker_count = %u\n", tracker_count);
        /* Get TDB information from context */
        uint32_t file_length_arg = file_length;
        uint32_t entry_point = (tinfo && tinfo->status == 1) ? tinfo->exeinfo.ep : 0;
        uint32_t num_sections = (tinfo && tinfo->status == 1) ? tinfo->exeinfo.nsections : 0;
        uint32_t container_type = (ctx && ctx->recursion_level >= 0) ? 
                                cli_recursion_stack_get_type(ctx, -1) : 0;
        uint32_t num_intermediates = 0;
        uint32_t is_pe_target = (tinfo && tinfo->status == 1) ? 1 : 0;
        uint32_t expr_bytecode_size = m->expr_bytecode_size;

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
        int a2 = 0;
        clSetKernelArg(rt->lsig_kernel, a2++, sizeof(cl_mem), &m->d_lsig_metas);
        clSetKernelArg(rt->lsig_kernel, a2++, sizeof(uint32_t), &m->num_lsigs);
        clSetKernelArg(rt->lsig_kernel, a2++, sizeof(cl_mem), &m->d_expr_bytecode);
        clSetKernelArg(rt->lsig_kernel, a2++, sizeof(uint32_t), &expr_bytecode_size);
        clSetKernelArg(rt->lsig_kernel, a2++, sizeof(cl_mem), &rt->d_tracker_pool);
        clSetKernelArg(rt->lsig_kernel, a2++, sizeof(uint32_t), &tracker_count);
        clSetKernelArg(rt->lsig_kernel, a2++, sizeof(uint32_t), &file_length_arg);
        clSetKernelArg(rt->lsig_kernel, a2++, sizeof(uint32_t), &entry_point);
        clSetKernelArg(rt->lsig_kernel, a2++, sizeof(uint32_t), &num_sections);
        clSetKernelArg(rt->lsig_kernel, a2++, sizeof(uint32_t), &container_type);
        clSetKernelArg(rt->lsig_kernel, a2++, sizeof(cl_mem), &d_intermediates);
        clSetKernelArg(rt->lsig_kernel, a2++, sizeof(uint32_t), &num_intermediates);
        clSetKernelArg(rt->lsig_kernel, a2++, sizeof(uint32_t), &is_pe_target);
        clSetKernelArg(rt->lsig_kernel, a2++, sizeof(cl_mem), &d_icon_data);
        clSetKernelArg(rt->lsig_kernel, a2++, sizeof(uint32_t), &icon_data_size);
        clSetKernelArg(rt->lsig_kernel, a2++, sizeof(cl_mem), &rt->d_result);
        clSetKernelArg(rt->lsig_kernel, a2++, sizeof(cl_mem), &m->d_virname_pool);
        
        size_t gws_lsig = ((m->num_lsigs + 255) / 256) * 256;
      
        clEnqueueNDRangeKernel(rt->queue, rt->lsig_kernel, 1, NULL,
                            &gws_lsig, NULL, 0, NULL, NULL);
        
        /* Wait for kernel to complete before cleaning up */
        clFinish(rt->queue);
        fflush(stdout);
        
        if (d_intermediates) clReleaseMemObject(d_intermediates);
        if (d_icon_data) clReleaseMemObject(d_icon_data);
    }
 
    clFinish(rt->queue);
    fflush(stdout);
    
    /* Read final result */
    gpu_scan_result_t h_result;
    clEnqueueReadBuffer(rt->queue, rt->d_result, CL_TRUE,
                        0, sizeof(gpu_scan_result_t), &h_result, 0, NULL, NULL);

    fprintf(stderr, "GPU RESULT: result_code=%d, virname_offset=%u, virname_len=%u\n",
            h_result.result_code, h_result.virname_offset, h_result.virname_len);

    if (h_result.needs_cpu_fallback) {
        fprintf(stderr, "GPU: Falling back to CPU due to needs_cpu_fallback flag being true %u\n", 
                h_result.fallback_offset);
        return GPU_RESULT_BREAK;
    }

    if (h_result.result_code == GPU_RESULT_VIRUS && h_result.virname_offset < m->virname_pool_size) {
        fprintf(stderr, "Virus name from pool: '%s'\n", 
                m->h_virname_pool + h_result.virname_offset);
    }
    
    if (h_result.result_code == GPU_RESULT_VIRUS) {

            if (h_result.virname_offset >= m->virname_pool_size) {
                fprintf(stderr, "ERROR: Invalid virname_offset=%u >= pool_size=%u\n",
                        h_result.virname_offset, m->virname_pool_size);
            } else {
                fprintf(stderr, "Virus: %s\n", m->h_virname_pool + h_result.virname_offset);
            }
        // fprintf(stderr, "[GPU] VIRUS DETECTED BY GPU (matcher %u): %s\n", 
        //         matcher_idx, m->h_virname_pool + h_result.virname_offset);
        *virname = m->h_virname_pool + h_result.virname_offset;
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

 

// cl_error_t gpu_collect_batch_results(struct gpu_rt *rt, struct cli_matcher *root,
//                                       cli_ctx *ctx, const char **virname)
// {
//     if (!rt->batch.active) return CL_SUCCESS;
    
//     cl_error_t final_result = CL_CLEAN;
    
//     for (uint32_t f = 0; f < rt->batch.count; f++) {
//         if (rt->batch.results[f] == CL_VIRUS) {
//             if (virname && rt->batch.virnames[f])
//                 *virname = rt->batch.virnames[f];
//             final_result = CL_VIRUS;
//             break;
//         }
//         if (rt->batch.results[f] == CL_BREAK) {
//             final_result = CL_BREAK;
//         }
//     }
    
//     /* Cleanup */
//     for (uint32_t i = 0; i < rt->batch.count; i++) {
//         if (rt->batch.buffers[i]) {
//             free((void*)rt->batch.buffers[i]);
//             rt->batch.buffers[i] = NULL;
//         }
//     }
    
//     free(rt->batch.hit_counts); rt->batch.hit_counts = NULL;
//     free(rt->batch.results); rt->batch.results = NULL;
//     free(rt->batch.virnames); rt->batch.virnames = NULL;
//     rt->batch.count = 0;
//     rt->batch.total_bytes = 0;
//     rt->batch.active = false;
    
//     return final_result;
// }

 

 

 


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
    cl_platform_id platforms[10];
    cl_uint num_platforms;
    cl_device_id devices[10];
    cl_uint num_devices;
    char platform_name[256];
    char device_name[256];
    int i, j;
    
    if (!rt) return -1;
    if (rt->initialized) return 0;
    
    /* Initialize all matcher slots to zero */
    for (i = 0; i < GPU_MAX_MATCHERS; i++) {
        memset(&rt->matchers[i], 0, sizeof(struct gpu_matcher_data));
    }
    
    /* Get all OpenCL platforms */
    err = clGetPlatformIDs(10, platforms, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        cli_errmsg("GPU: No OpenCL platforms found\n");
        return -1;
    }
    
    cli_dbgmsg("GPU: Found %u platforms\n", num_platforms);
    
    /* Find AMD platform with GPU devices */
    int found = 0;
    for (i = 0; i < num_platforms && !found; i++) {
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 
                         sizeof(platform_name), platform_name, NULL);
        cli_dbgmsg("GPU: Platform %d: %s\n", i, platform_name);
        
        /* Check for AMD platform */
        if (strstr(platform_name, "AMD") || strstr(platform_name, "Advanced")) {
            /* Try to get GPU devices */
            err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 
                                10, devices, &num_devices);
            if (err == CL_SUCCESS && num_devices > 0) {
                rt->platform = platforms[i];
                rt->device = devices[0];
                
                clGetDeviceInfo(rt->device, CL_DEVICE_NAME, 
                               sizeof(device_name), device_name, NULL);
                cli_dbgmsg("GPU: Selected AMD platform with device: %s\n", device_name);
                found = 1;
                break;
            }
        }
    }
    
    /* If no AMD GPU, try any GPU */
    if (!found) {
        for (i = 0; i < num_platforms && !found; i++) {
            err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 
                                10, devices, &num_devices);
            if (err == CL_SUCCESS && num_devices > 0) {
                rt->platform = platforms[i];
                rt->device = devices[0];
                
                clGetPlatformInfo(rt->platform, CL_PLATFORM_NAME, 
                                 sizeof(platform_name), platform_name, NULL);
                clGetDeviceInfo(rt->device, CL_DEVICE_NAME, 
                               sizeof(device_name), device_name, NULL);
                cli_dbgmsg("GPU: Selected platform %s with device: %s\n", 
                          platform_name, device_name);
                found = 1;
                break;
            }
        }
    }
    
    if (!found) {
        cli_errmsg("GPU: No GPU devices found\n");
        return -1;
    }

    /* Create OpenCL context */
    rt->context = clCreateContext(NULL, 1, &rt->device, NULL, NULL, &err);
    if (!rt->context) {
        cli_errmsg("GPU: Failed to create context: %d\n", err);
        return -1;
    }

    /* Create command queue */
    rt->queue = clCreateCommandQueue(rt->context, rt->device, 0, &err);
    if (!rt->queue) {
        cli_errmsg("GPU: Failed to create command queue: %d\n", err);
        clReleaseContext(rt->context);
        return -1;
    }

    /* Set scan buffer size */
    rt->scan_buffer_size = 256 * 1024 * 1024; /* 256MB */
    
    /* ========== CREATE ALL NECESSARY BUFFERS ========== */
    
    /* Scan buffer - for file data */
    rt->scan_buffer = clCreateBuffer(rt->context, CL_MEM_READ_WRITE,
                                     rt->scan_buffer_size, NULL, &err);
    if (err != CL_SUCCESS || !rt->scan_buffer) {
        cli_errmsg("GPU: Failed to create scan buffer: %d\n", err);
        goto error;
    }
    fprintf(stderr, "GPU: Created scan_buffer (%u MB)\n", rt->scan_buffer_size / (1024*1024));
    
    /* Tracker pool - for multi-part pattern matching (CRITICAL) */
    rt->d_tracker_pool = clCreateBuffer(rt->context, CL_MEM_READ_WRITE,
                                        GPU_MAX_TRACKERS * sizeof(gpu_multipart_tracker_t),
                                        NULL, &err);
    if (err != CL_SUCCESS || !rt->d_tracker_pool) {
        cli_errmsg("GPU: Failed to create tracker pool buffer: %d\n", err);
        goto error;
    }
    fprintf(stderr, "GPU: Created d_tracker_pool buffer\n");
    
    /* Tracker count (CRITICAL) */
    rt->d_tracker_count = clCreateBuffer(rt->context, CL_MEM_READ_WRITE,
                                         sizeof(uint32_t), NULL, &err);
    if (err != CL_SUCCESS || !rt->d_tracker_count) {
        cli_errmsg("GPU: Failed to create tracker count buffer: %d\n", err);
        goto error;
    }
    fprintf(stderr, "GPU: Created d_tracker_count buffer\n");
    
    /* Result buffer (CRITICAL) */
    rt->d_result = clCreateBuffer(rt->context, CL_MEM_READ_WRITE,
                                  sizeof(gpu_scan_result_t), NULL, &err);
    if (err != CL_SUCCESS || !rt->d_result) {
        cli_errmsg("GPU: Failed to create result buffer: %d\n", err);
        goto error;
    }
    fprintf(stderr, "GPU: Created d_result buffer\n");
    
    /* Get device info for debugging */
    cl_ulong max_mem_alloc_size;
    cl_ulong global_mem_size;
    cl_uint max_compute_units;
    
    clGetDeviceInfo(rt->device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, 
                    sizeof(max_mem_alloc_size), &max_mem_alloc_size, NULL);
    clGetDeviceInfo(rt->device, CL_DEVICE_GLOBAL_MEM_SIZE, 
                    sizeof(global_mem_size), &global_mem_size, NULL);
    clGetDeviceInfo(rt->device, CL_DEVICE_MAX_COMPUTE_UNITS, 
                    sizeof(max_compute_units), &max_compute_units, NULL);
    
    fprintf(stderr, "GPU Device Info:\n");
    fprintf(stderr, "  Global Memory: %llu MB\n", (unsigned long long)global_mem_size / (1024*1024));
    fprintf(stderr, "  Max Allocation: %llu MB\n", (unsigned long long)max_mem_alloc_size / (1024*1024));
    fprintf(stderr, "  Compute Units: %u\n", max_compute_units);
    
    /* Build kernel from source */
    FILE *fp = fopen("../libclamav/gpu_ac_kernel.cl", "r");
    if (!fp) {
        cli_errmsg("GPU: Failed to open kernel file\n");
        goto error;
    }

    fseek(fp, 0, SEEK_END);
    long kernel_len = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    char *kernel_src = malloc(kernel_len + 1);
    fread(kernel_src, 1, kernel_len, fp);
    kernel_src[kernel_len] = '\0';
    fclose(fp);

    rt->program = clCreateProgramWithSource(rt->context, 1, 
                                            (const char **)&kernel_src, 
                                            &kernel_len, &err);
    free(kernel_src);
    if (err != CL_SUCCESS || !rt->program) {
        cli_errmsg("GPU: Failed to create program from source: %d\n", err);
        goto error;
    }
    
    /* Build with optimization options */
    const char *build_opts = "-cl-std=CL1.2 -cl-mad-enable -cl-fast-relaxed-math";
    err = clBuildProgram(rt->program, 1, &rt->device, build_opts, NULL, NULL);
    
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(rt->program, rt->device, CL_PROGRAM_BUILD_LOG,
                              0, NULL, &log_size);
        char *log = malloc(log_size + 1);
        if (log) {
            clGetProgramBuildInfo(rt->program, rt->device, CL_PROGRAM_BUILD_LOG,
                                  log_size, log, NULL);
            log[log_size] = 0;
            cli_errmsg("GPU kernel build failed:\n%s\n", log);
            free(log);
        }
        goto error;
    }
    
    /* Create main kernel */
    rt->kernel = clCreateKernel(rt->program, "ac_scan_validate", &err);
    if (err != CL_SUCCESS || !rt->kernel) {
        cli_errmsg("GPU: Failed to create main kernel: %d\n", err);
        goto error;
    }
    fprintf(stderr, "GPU: Created main kernel\n");
    
    /* Create logical signature kernel (optional) */
    rt->lsig_kernel = clCreateKernel(rt->program, "evaluate_logical_sigs", &err);
    if (err != CL_SUCCESS || !rt->lsig_kernel) {
        cli_dbgmsg("GPU: Logical signature kernel not available (%d)\n", err);
        rt->lsig_kernel = NULL;
        /* Not fatal - continue without logical kernel */
    } else {
        fprintf(stderr, "GPU: Created logical signature kernel\n");
    }
    
    /* Initialize batch processing structures */
    rt->batch.max_count = 64;
    rt->batch.buffers = malloc(rt->batch.max_count * sizeof(void*));
    rt->batch.lengths = malloc(rt->batch.max_count * sizeof(uint32_t));
    rt->batch.file_offsets = malloc(rt->batch.max_count * sizeof(uint32_t));
    rt->batch.kernel_events = NULL;
    rt->batch.count_events = NULL;
    rt->batch.hit_counts = NULL;
    rt->batch.count = 0;
    rt->batch.total_bytes = 0;
    rt->batch.active = false;
    
    if (!rt->batch.buffers || !rt->batch.lengths || !rt->batch.file_offsets) {
        cli_errmsg("GPU: Failed to allocate batch structures\n");
        goto error;
    }
    
    /* Initialize global state */
    rt->dfa_uploaded = 0;
    rt->num_lsigs = 0;
    rt->num_gpu_patterns = 0;
    rt->virname_pool_size = 0;
    rt->h_virname_pool = NULL;
    rt->initialized = 1;
    
    fprintf(stderr, "GPU: gpu_rt_init SUCCESS\n");
    return 0;
    
error:
    /* Cleanup on error */
    if (rt->scan_buffer) { clReleaseMemObject(rt->scan_buffer); rt->scan_buffer = NULL; }
    if (rt->d_tracker_pool) { clReleaseMemObject(rt->d_tracker_pool); rt->d_tracker_pool = NULL; }
    if (rt->d_tracker_count) { clReleaseMemObject(rt->d_tracker_count); rt->d_tracker_count = NULL; }
    if (rt->d_result) { clReleaseMemObject(rt->d_result); rt->d_result = NULL; }
    if (rt->kernel) { clReleaseKernel(rt->kernel); rt->kernel = NULL; }
    if (rt->lsig_kernel) { clReleaseKernel(rt->lsig_kernel); rt->lsig_kernel = NULL; }
    if (rt->program) { clReleaseProgram(rt->program); rt->program = NULL; }
    if (rt->queue) { clReleaseCommandQueue(rt->queue); rt->queue = NULL; }
    if (rt->context) { clReleaseContext(rt->context); rt->context = NULL; }
    
    if (rt->batch.buffers) { free(rt->batch.buffers); rt->batch.buffers = NULL; }
    if (rt->batch.lengths) { free(rt->batch.lengths); rt->batch.lengths = NULL; }
    if (rt->batch.file_offsets) { free(rt->batch.file_offsets); rt->batch.file_offsets = NULL; }
    
    return -1;
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
    flat->out_count = cli_calloc(num_states, sizeof(uint32_t));
    
    fprintf(stderr, "STEP 5: Allocated tables: next=%p, out_index=%p, out_count=%p\n", 
            flat->next, flat->out_index, flat->out_count);
    
    if (!flat->next || !flat->out_index || !flat->out_count) {
        cli_errmsg("GPU-DFA: Failed to allocate flattened DFA tables\n");
        goto error;
    }
    
    /* Count total outputs - NOW including entire failure chain */
    uint32_t total_out = 0;
    for (uint32_t s = 0; s < num_states; s++) {
        struct cli_ac_node *node = states[s];
        
        /* Count outputs from node and ALL failure ancestors */
        struct cli_ac_node *fail_node = node;
        while (fail_node) {
            for (struct cli_ac_list *l = fail_node->list; l; l = l->next) {
                total_out++;
            }
            fail_node = fail_node->fail;
        }
    }
    fprintf(stderr, "STEP 6: total_out=%u\n", total_out);

    if (total_out == 0) {
        cli_errmsg("GPU-DFA: No patterns found! Check if signatures are loaded.\n");
        goto error;
    }
 
    flat->out_pat = cli_malloc(total_out * sizeof(uint32_t));  // Changed to uint32_t
    flat->sig_id = cli_calloc(total_out, sizeof(uint32_t));
    flat->part_no = cli_calloc(total_out, sizeof(uint32_t));
    fprintf(stderr, "STEP 7: Allocated output arrays\n");

    if (!flat->out_pat || !flat->sig_id || !flat->part_no) {
        cli_errmsg("GPU-DFA: Failed to allocate %u pattern tables\n", total_out);
        goto error;
    }
    
    /* Build transition table - this is correct */
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
    }
    for (uint32_t state = 0; state < num_states; state++) {
    struct cli_ac_node *node = states[state];
    for (uint32_t c = 0; c < 256; c++) {
        if (flat->next[state * 256 + c] == 0 && node->fail) {
            flat->next[state * 256 + c] = flat->next[node->fail->gpu_state_id * 256 + c];
        }
    }
}
    fprintf(stderr, "STEP 9: Built transition table\n");
    
    /* Fill pattern tables with metadata - NOW including ALL failure ancestors */
    uint32_t pat_idx = 0;
    uint32_t max_pattern_id = 0;
    for (uint32_t s = 0; s < num_states; s++) {
        struct cli_ac_node *node = states[s];
        for (struct cli_ac_list *l = node->list; l; l = l->next) {
            if (l->me->gpu_id > max_pattern_id)
                max_pattern_id = l->me->gpu_id;
        }
    }
    fprintf(stderr, "STEP 10: max_pattern_id=%u\n", max_pattern_id);

    uint8_t *seen_bitmap = NULL; 
    seen_bitmap = cli_calloc((max_pattern_id + 7) / 8, sizeof(uint8_t));
    if (!seen_bitmap) goto error;
    fprintf(stderr, "STEP 11: Allocated bitmap\n");
    
    for (uint32_t state = 0; state < num_states; state++) {
        struct cli_ac_node *node = states[state];
        
        memset(seen_bitmap, 0, (max_pattern_id + 7) / 8);
        
        flat->out_index[state] = pat_idx;
        
        /* CRITICAL: Collect patterns from node and ALL failure ancestors */
        struct cli_ac_node *fail_node = node;
        while (fail_node) {
            for (struct cli_ac_list *l = fail_node->list; l; l = l->next) {
                struct cli_ac_patt *patt = l->me;
                if (patt && patt->gpu_id < flat->pat_count) {
                    uint32_t byte = patt->gpu_id >> 3;
                    uint8_t bit = 1 << (patt->gpu_id & 7);
                    if (!(seen_bitmap[byte] & bit)) {
                        seen_bitmap[byte] |= bit;
                        flat->out_pat[pat_idx] = patt->gpu_id;
                        flat->sig_id[pat_idx] = patt->sigid;
                        flat->part_no[pat_idx] = patt->partno;
                        flat->out_count[state]++;
                        pat_idx++;
                    }
                }
            }
            fail_node = fail_node->fail;
        }
        
        // if (pat_idx > flat->out_index[state]) {
        //     fprintf(stderr, "STATE %u: %u outputs\n", state, flat->out_count[state]);
        // }
    }
    flat->out_total = pat_idx;

    /* Validate output pattern IDs */
    uint32_t max_pat_id = 0;
    for (uint32_t i = 0; i < flat->out_total; i++) {
        if (flat->out_pat[i] > max_pat_id) max_pat_id = flat->out_pat[i];
    }
    fprintf(stderr, "Output pattern IDs: min=0, max=%u, pat_count=%u\n", 
            max_pat_id, flat->pat_count);
    
    /* Debug first few out_pat values */
    fprintf(stderr, "DEBUG: First 10 out_pat values:\n");
    for (uint32_t i = 0; i < 10 && i < flat->out_total; i++) {
        fprintf(stderr, "  out_pat[%u] = %u\n", i, flat->out_pat[i]);
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



int gpu_rt_upload_flattened_dfa(struct gpu_rt *rt, struct gpu_flattened_dfa *dfa, uint32_t matcher_idx)
{
    cl_int err;
    struct gpu_matcher_data *m = &rt->matchers[matcher_idx];
    
    if (!rt || !dfa) return -1;
    
    /* Clear existing DFA buffers for this matcher */
    if (m->dfa_next) {
        clReleaseMemObject(m->dfa_next);
        m->dfa_next = NULL;
    }
    if (m->dfa_out_index) {
        clReleaseMemObject(m->dfa_out_index);
        m->dfa_out_index = NULL;
    }
    if (m->dfa_out_count) {
        clReleaseMemObject(m->dfa_out_count);
        m->dfa_out_count = NULL;
    }
    if (m->dfa_out_pat) {
        clReleaseMemObject(m->dfa_out_pat);
        m->dfa_out_pat = NULL;
    }
    if (m->dfa_sig_id) {
        clReleaseMemObject(m->dfa_sig_id);
        m->dfa_sig_id = NULL;
    }
    if (m->dfa_part_no) {
        clReleaseMemObject(m->dfa_part_no);
        m->dfa_part_no = NULL;
    }
    
    fprintf(stderr, "=== DFA UPLOAD for matcher %u ===\n", matcher_idx);
    fprintf(stderr, "states=%u, out_total=%u, pat_count=%u\n", 
            dfa->states, dfa->out_total, dfa->pat_count);
    
    /* Calculate buffer sizes */
    size_t next_size = (size_t)dfa->states * 256 * sizeof(uint32_t);
    size_t index_size = (size_t)dfa->states * sizeof(uint32_t);
    size_t count_size = (size_t)dfa->states * sizeof(uint16_t);
    size_t pat_size = (size_t)dfa->out_total * sizeof(uint32_t);
    size_t sig_size = (size_t)dfa->out_total * sizeof(uint32_t);
    size_t part_size = (size_t)dfa->out_total * sizeof(uint32_t);
    
    fprintf(stderr, "next_size=%zu bytes (%zu MB)\n", next_size, next_size/(1024*1024));
    
    /* Create buffers - store in matcher slot */
    m->dfa_next = clCreateBuffer(rt->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  next_size, dfa->next, &err);
    if (err != CL_SUCCESS || !m->dfa_next) {
        fprintf(stderr, "Failed to create dfa_next buffer: %d\n", err);
        goto error;
    }
    
    m->dfa_out_index = clCreateBuffer(rt->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       index_size, dfa->out_index, &err);
    if (err != CL_SUCCESS || !m->dfa_out_index) goto error;
    
    m->dfa_out_count = clCreateBuffer(rt->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       count_size, dfa->out_count, &err);
    if (err != CL_SUCCESS || !m->dfa_out_count) goto error;
    
    m->dfa_out_pat = clCreateBuffer(rt->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     pat_size, dfa->out_pat, &err);
    if (err != CL_SUCCESS || !m->dfa_out_pat) goto error;
    
    m->dfa_sig_id = clCreateBuffer(rt->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sig_size, dfa->sig_id, &err);
    if (err != CL_SUCCESS || !m->dfa_sig_id) goto error;
    
    m->dfa_part_no = clCreateBuffer(rt->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     part_size, dfa->part_no, &err);
    if (err != CL_SUCCESS || !m->dfa_part_no) goto error;
    
    /* Store metadata in matcher slot */
    m->dfa_states = dfa->states;
    m->out_total = dfa->out_total;
    m->pat_count = dfa->pat_count;
    m->uploaded = 1;

    rt->dfa_uploaded = 1;
    rt->dfa_states = dfa->states;
    rt->out_total = dfa->out_total;
    
    fprintf(stderr, "=== DFA UPLOAD SUCCESS for matcher %u ===\n", matcher_idx);
    return 0;
    
error:
    fprintf(stderr, "DFA upload failed for matcher %u\n", matcher_idx);
    if (m->dfa_next) { clReleaseMemObject(m->dfa_next); m->dfa_next = NULL; }
    if (m->dfa_out_index) { clReleaseMemObject(m->dfa_out_index); m->dfa_out_index = NULL; }
    if (m->dfa_out_count) { clReleaseMemObject(m->dfa_out_count); m->dfa_out_count = NULL; }
    if (m->dfa_out_pat) { clReleaseMemObject(m->dfa_out_pat); m->dfa_out_pat = NULL; }
    if (m->dfa_sig_id) { clReleaseMemObject(m->dfa_sig_id); m->dfa_sig_id = NULL; }
    if (m->dfa_part_no) { clReleaseMemObject(m->dfa_part_no); m->dfa_part_no = NULL; }
    m->uploaded = 0;
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
 
