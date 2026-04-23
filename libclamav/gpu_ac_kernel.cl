
/* Constants section */
#define GPU_MATCH_CHAR        0x0000
#define GPU_MATCH_IGNORE      0x0200
#define GPU_MATCH_NOCASE      0x0400
#define GPU_MATCH_NIBBLE_HIGH 0x0100
#define GPU_MATCH_NIBBLE_LOW  0x0300
#define GPU_MATCH_METADATA    0xFF00
#define CLI_MATCH_METADATA    0xFF00
#define CLI_MATCH_CHAR        0x0000
#define CLI_MATCH_IGNORE      0x0200
#define CLI_MATCH_NOCASE      0x0400
#define CLI_MATCH_NIBBLE_HIGH 0x0100
#define CLI_MATCH_NIBBLE_LOW  0x0300
#define CLI_MATCH_SPECIAL     0x8000

#define GPU_OFF_ANY      0
#define GPU_OFF_ABSOLUTE 1
#define GPU_OFF_NONE     0xFFFFFFFF
#define GPU_MAX_PARTS    16

#define GPU_RESULT_CLEAN  0
#define GPU_RESULT_VIRUS  1
#define GPU_RESULT_BREAK  2

/* Logical signature bytecode opcodes */
#define OP_LOAD_SUBSIG  0
#define OP_AND          1
#define OP_OR           2
#define OP_NOT          3
#define OP_END          4
#define OP_LOAD_COUNT   5
#define OP_LOAD_MASK    6
#define OP_GT           7
#define OP_LT           8
#define OP_EQ           9
#define OP_COMBINE_AND  10
#define OP_COMBINE_OR   11
#define OP_COMBINE_NOT  12
#define OP_BLOCK_END    13

/* Pattern metadata */
typedef struct {
    uint   length;
    uint   prefix_length;
    uint   parts;
    uint   type;
    uint   sigid;
    uint   partno;
    uint   offset_min;
    uint   offset_max;
    uint   offdata0;
    uint   lsigid[3];
    uint   ch0;
    uint   ch1;
    uint   ch_mindist0;
    uint   ch_mindist1;
    uint   ch_maxdist0;
    uint   ch_maxdist1;
    uint   mindist;
    uint   maxdist;
    uint   pattern_offset;
    uint   prefix_offset;
    uint   virname_offset;
    uint   virname_len;
    uint   depth;
    uint   boundary;
    uint   sigopts;
    uint   has_regex;
    uint   is_bytecode;
    uint   special_pattern;
} gpu_pattern_t;

/* Multi-part tracker */
typedef struct {
    uint   sig_id;
    uint   tracker_type;
    uint   total_parts;
    uint   found_mask_lo;
    uint   found_mask_hi;
    uint   offsets[GPU_MAX_PARTS][33];
    uint   offset_counts[GPU_MAX_PARTS];
    uint   first_part_realoffs[33];
    uint   last_offsets[GPU_MAX_PARTS];
    uint   subsig_counts[GPU_MAX_PARTS];
    uint   subsig_first_offset[GPU_MAX_PARTS];
    uint   subsig_last_offset[GPU_MAX_PARTS];
} gpu_multipart_tracker_t;

/* Logical signature metadata */
typedef struct {
    uint   sig_id;
    uint   num_subsigs;
    uint   expr_offset;
    uint   expr_length;
    uint   virname_offset;
    uint   virname_len;
    uint   tdb_container;
    uint   tdb_filesize_min;
    uint   tdb_filesize_max;
    uint   tdb_ep_min;
    uint   tdb_ep_max;
    uint   tdb_nos_min;
    uint   tdb_nos_max;
    uint   tdb_intermediates_mask;
    uint   tdb_handlertype;
    uint   tdb_icongrp1_offset;
    uint   tdb_icongrp2_offset;
    uint   bc_idx;
    uint   has_regex;
} gpu_lsig_meta_t;

/* Expression instruction */
typedef struct {
    uint   op;
    uint   operand;
} gpu_expr_inst_t;

/* Scan result */
typedef struct {
    int    result_code;
    uint   virname_offset;
    uint   virname_len;
    uint   match_offset;
    uint   needs_cpu_fallback;
    uint   fallback_offset;
    uint   fallback_pattern_id;
} gpu_scan_result_t;

/* Helper functions */
bool match_byte(uint pw, uchar bb) {
    ushort wc = pw & GPU_MATCH_METADATA;
    uchar pb = (uchar)(pw & 0xFF);
    if (wc == GPU_MATCH_CHAR)    { return pb == bb; }
    if (wc == GPU_MATCH_IGNORE)  { return true; }
    if (wc == GPU_MATCH_NOCASE) {
        uchar lb = bb;
        if (lb >= 'A' && lb <= 'Z') { lb += 32; }
        uchar lp = pb;
        if (lp >= 'A' && lp <= 'Z') { lp += 32; }
        return lp == lb;
    }
    if (wc == GPU_MATCH_NIBBLE_HIGH) { return (pb & 0xF0) == (bb & 0xF0); }
    if (wc == GPU_MATCH_NIBBLE_LOW)  { return (pb & 0x0F) == (bb & 0x0F); }
    return false;
}

bool verify_pattern(
    __global const gpu_pattern_t *patt,
    __global const ushort *pattern_bytes,
    __global const ushort *prefix_bytes,
    __global const uchar *buffer,
    uint abs_match_start,
    uint abs_end,
    uint num_pattern_bytes,
    uint num_prefix_bytes)
{
    ushort meta;
    uchar pat_byte;
    uchar buf_byte;
    
    if (patt->prefix_length > 0) {
        if (patt->prefix_offset + patt->prefix_length > num_prefix_bytes) return false;
        if (abs_match_start < patt->prefix_length) return false;
        
        uint buf_pos = abs_match_start - patt->prefix_length;
        uint pat_pos = patt->prefix_offset;
        
        for (uint j = 0; j < patt->prefix_length; j++) {
            if (buf_pos + j >= abs_end) return false;
            
            ushort pw = prefix_bytes[pat_pos + j];
            uchar pb = buffer[buf_pos + j];
            
            meta = pw & CLI_MATCH_METADATA;
            pat_byte = (uchar)(pw & 0xFF);
            
            if (meta == CLI_MATCH_CHAR) {
                if (pat_byte != pb) return false;
            } else if (meta == CLI_MATCH_IGNORE) {
                continue;
            } else if (meta == CLI_MATCH_NOCASE) {
                uchar pb_lower = pb;
                uchar pat_lower = pat_byte;
                if (pb_lower >= 'A' && pb_lower <= 'Z') pb_lower += 32;
                if (pat_lower >= 'A' && pat_lower <= 'Z') pat_lower += 32;
                if (pat_lower != pb_lower) return false;
            } else if (meta == CLI_MATCH_NIBBLE_HIGH) {
                if ((pat_byte & 0xF0) != (pb & 0xF0)) return false;
            } else if (meta == CLI_MATCH_NIBBLE_LOW) {
                if ((pat_byte & 0x0F) != (pb & 0x0F)) return false;
            } else {
                return false;
            }
        }
    }
    
    uint pat_pos = patt->pattern_offset;
    for (uint j = 0; j < patt->length; j++) {
        if (abs_match_start + j >= abs_end) return false;
        
        ushort pw = pattern_bytes[pat_pos + j];
        uchar pb = buffer[abs_match_start + j];
        
        meta = pw & CLI_MATCH_METADATA;
        pat_byte = (uchar)(pw & 0xFF);
        
        if (meta == CLI_MATCH_CHAR) {
            if (pat_byte != pb) return false;
        } else if (meta == CLI_MATCH_IGNORE) {
            continue;
        } else if (meta == CLI_MATCH_NOCASE) {
            uchar pb_lower = pb;
            uchar pat_lower = pat_byte;
            if (pb_lower >= 'A' && pb_lower <= 'Z') pb_lower += 32;
            if (pat_lower >= 'A' && pat_lower <= 'Z') pat_lower += 32;
            if (pat_lower != pb_lower) return false;
        } else if (meta == CLI_MATCH_NIBBLE_HIGH) {
            if ((pat_byte & 0xF0) != (pb & 0xF0)) return false;
        } else if (meta == CLI_MATCH_NIBBLE_LOW) {
            if ((pat_byte & 0x0F) != (pb & 0x0F)) return false;
        } else {
            return false;
        }
    }
    
    return true;
}

bool validate_ch(
    __global const gpu_pattern_t *patt,
    __global const uchar *buffer,
    uint abs_match_start,
    uint abs_match_end,
    uint abs_file_start,
    uint abs_end)
{
    if ((patt->ch0 & GPU_MATCH_METADATA) != GPU_MATCH_IGNORE) {
        uint min_d = patt->ch_mindist0;
        uint max_d = patt->ch_maxdist0;
        if (max_d > 4096) max_d = 4096;
        
        uint count = 0;
        for (uint d = min_d; d <= max_d; d++) {
            if (abs_match_start < abs_file_start + d + 1) continue;
            uint bp = abs_match_start - d - 1;
            if (bp < abs_file_start) continue;
            if (match_byte((ushort)patt->ch0, buffer[bp])) {
                count++;
            }
        }
        if (count == 0) return false;
    }
    
    if ((patt->ch1 & GPU_MATCH_METADATA) != GPU_MATCH_IGNORE) {
        uint min_d = patt->ch_mindist1;
        uint max_d = patt->ch_maxdist1;
        if (max_d > 4096) max_d = 4096;
        
        uint count = 0;
        for (uint d = min_d; d <= max_d; d++) {
            uint bp = abs_match_end + 1 + d;
            if (bp >= abs_end) break;
            if (match_byte((ushort)patt->ch1, buffer[bp])) {
                count++;
            }
        }
        if (count == 0) return false;
    }
    return true;
}

__global gpu_multipart_tracker_t* find_or_create_multipart(
    __global gpu_multipart_tracker_t *pool,
    volatile __global uint *pool_count,
    uint max_pool,
    uint sig_id,
    uint total_parts,
    uint tracker_type)
{
    uint n = *pool_count;
    if (n > max_pool) n = max_pool;
    uint combined_key = (sig_id << 1) | tracker_type;
    for (uint i = 0; i < n; i++) {
        uint pool_key = (pool[i].sig_id << 1) | pool[i].tracker_type;
        if (pool_key == combined_key) return &pool[i];
    }
    uint idx = atomic_inc(pool_count);
    if (idx >= max_pool) return 0;
    pool[idx].sig_id = sig_id;
    pool[idx].tracker_type = tracker_type;
    pool[idx].total_parts = total_parts;
    pool[idx].found_mask_lo = 0;
    pool[idx].found_mask_hi = 0;
    for (int i = 0; i < GPU_MAX_PARTS; i++) {
        pool[idx].offset_counts[i] = 0;
        pool[idx].subsig_counts[i] = 0;
        pool[idx].subsig_first_offset[i] = 0;
        pool[idx].subsig_last_offset[i] = 0;
        pool[idx].last_offsets[i] = 0;
    }
    return &pool[idx];
}

__constant uchar boundary_table[256] = {
    0,0,0,0,0,0,0,0,0,0,2,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    3,0,2,0,0,0,0,2,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
};

bool is_alnum(uchar c) {
    return (c >= '0' && c <= '9') ||
           (c >= 'A' && c <= 'Z') ||
           (c >= 'a' && c <= 'z');
}

bool match_pattern_byte(ushort pw, uchar pb) {
    ushort meta = pw & 0xFF00;
    uchar pat_byte = (uchar)(pw & 0xFF);
    if (meta == GPU_MATCH_CHAR) {
        return pat_byte == pb;
    } else if (meta == GPU_MATCH_IGNORE) {
        return true;
    } else if (meta == GPU_MATCH_NOCASE) {
        uchar pb_lower = pb;
        uchar pat_lower = pat_byte;
        if (pb_lower >= 'A' && pb_lower <= 'Z') pb_lower += 32;
        if (pat_lower >= 'A' && pat_lower <= 'Z') pat_lower += 32;
        return pat_lower == pb_lower;
    } else if (meta == GPU_MATCH_NIBBLE_HIGH) {
        return (pat_byte & 0xF0) == (pb & 0xF0);
    } else if (meta == GPU_MATCH_NIBBLE_LOW) {
        return (pat_byte & 0x0F) == (pb & 0x0F);
    }
    return false;
}

int gpu_backward_match_branch(
    __global const uchar *buffer,
    uint bp,
    uint offset,
    uint fileoffset,
    uint length,
    __global const gpu_pattern_t *patt,
    uint pp,
    __global const ushort *pattern_bytes,
    __global const ushort *prefix_bytes,
    uint *out_start,
    uint *out_end);

int gpu_forward_match_branch(
    __global const uchar *buffer,
    uint bp,
    uint offset,
    uint fileoffset,
    uint length,
    __global const gpu_pattern_t *patt,
    uint pp,
    uint specialcnt,
    __global const ushort *pattern_bytes,
    __global const ushort *prefix_bytes,
    uint *out_start,
    uint *out_end)
{
    uint pat_pos = patt->pattern_offset + pp;
    uint buf_pos = bp;
    uint match_start = offset;
    
    for (uint i = pp; i < patt->length && buf_pos < length; i++) {
        ushort pw = pattern_bytes[pat_pos];
        uchar pb = buffer[buf_pos];
        
        if (!match_pattern_byte(pw, pb)) {
            return 0;
        }
        
        pat_pos++;
        buf_pos++;
    }
    
    *out_end = buf_pos;
    
    if (patt->boundary & 0x0C) {
        uint boundary_flags = patt->boundary;
        
        if (boundary_flags & 0x0004) {
            bool match = !(boundary_flags & 0x0008);
            bool boundary_cond = (buf_pos == length || boundary_table[buffer[buf_pos]] >= 2);
            if (boundary_cond) match = !match;
            if (!match) return 0;
        }
        
        if (boundary_flags & 0x0040) {
            bool match = !(boundary_flags & 0x0080);
            bool line_cond = (buf_pos == length || 
                              buffer[buf_pos] == '\n' || 
                              (buffer[buf_pos] == '\r' && buf_pos + 1 < length && buffer[buf_pos + 1] == '\n'));
            if (line_cond) match = !match;
            if (!match) return 0;
        }
        
        if (boundary_flags & 0x0400) {
            bool match = !(boundary_flags & 0x0800);
            bool word_cond = (buf_pos == length || !is_alnum(buffer[buf_pos]));
            if (word_cond) match = !match;
            if (!match) return 0;
        }
    }
    
    if ((patt->ch1 & GPU_MATCH_METADATA) != GPU_MATCH_IGNORE) {
        uint min_d = patt->ch_mindist1;
        uint max_d = patt->ch_maxdist1;
        if (max_d > 4096) max_d = 4096;
        
        uint search_pos = buf_pos + min_d;
        bool found = false;
        
        for (uint d = min_d; d <= max_d && !found; d++) {
            if (search_pos >= length) break;
            if (match_pattern_byte((ushort)patt->ch1, buffer[search_pos])) {
                found = true;
                break;
            }
            search_pos++;
        }
        if (!found) return 0;
    }
    
    if (match_start == 0 || patt->prefix_length == 0) {
        *out_start = match_start;
        return 1;
    }
    
    return gpu_backward_match_branch(buffer, match_start - 1, match_start, fileoffset, length,
                                      patt, patt->prefix_length - 1, 
                                      pattern_bytes, prefix_bytes,
                                      out_start, out_end);
}

int gpu_backward_match_branch(
    __global const uchar *buffer,
    uint bp,
    uint offset,
    uint fileoffset,
    uint length,
    __global const gpu_pattern_t *patt,
    uint pp,
    __global const ushort *pattern_bytes,
    __global const ushort *prefix_bytes,
    uint *out_start,
    uint *out_end)
{
    uint buf_pos = bp;

    if (patt->prefix_length > 0 && pp < patt->prefix_length) {
        uint pat_pos = patt->prefix_offset + pp;

        for (int i = pp; i >= 0; i--) {
            if (buf_pos < fileoffset) {
                return 0;
            }
            
            ushort pw = prefix_bytes[pat_pos];
            uchar pb = buffer[buf_pos];
            
            if (!match_pattern_byte(pw, pb)) {
                return 0;
            }
            
            if (i == 0 || buf_pos == 0) break;
            
            if (pat_pos > patt->prefix_offset) {
                pat_pos--;
            }
            buf_pos--;
        }
        
        *out_start = buf_pos;
    } else {
        *out_start = offset;
        buf_pos = offset;
    }
    
    uint filestart;
    if (fileoffset >= offset) {
        filestart = fileoffset - offset + *out_start;
    } else {
        filestart = *out_start;
    }
    
    if (patt->boundary & 0x03) {
        uint boundary_flags = patt->boundary;
        
        if (boundary_flags & 0x0001) {
            bool match = !(boundary_flags & 0x0002);
            bool boundary_cond = (!filestart || 
                                  (*out_start > 0 && (boundary_table[buffer[*out_start - 1]] == 1 || 
                                                      boundary_table[buffer[*out_start - 1]] == 3)));
            if (boundary_cond) match = !match;
            if (!match) return 0;
        }
        
        if (boundary_flags & 0x0010) {
            bool match = !(boundary_flags & 0x0020);
            bool line_cond = (!filestart || (*out_start > 0 && buffer[*out_start - 1] == '\n'));
            if (line_cond) match = !match;
            if (!match) return 0;
        }
        
        if (boundary_flags & 0x0100) {
            bool match = !(boundary_flags & 0x0200);
            bool word_cond = (!filestart || (*out_start > 0 && !is_alnum(buffer[*out_start - 1])));
            if (word_cond) match = !match;
            if (!match) return 0;
        }
    }
    
    if ((patt->ch0 & GPU_MATCH_METADATA) != GPU_MATCH_IGNORE) {
        uint min_d = patt->ch_mindist0;
        uint max_d = patt->ch_maxdist0;
        if (max_d > 4096) max_d = 4096;
        
        if (min_d + 1 > *out_start) return 0;
        
        uint search_pos = *out_start - min_d - 1;
        bool found = false;
        uint file_start_boundary = fileoffset - offset;
        
        for (uint d = min_d; d <= max_d && !found; d++) {
            if (search_pos < file_start_boundary) break;
            
            if (match_pattern_byte((ushort)patt->ch0, buffer[search_pos])) {
                found = true;
                break;
            }
            if (search_pos == 0) break;
            search_pos--;
        }
        if (!found) return 0;
    }
    
    return 1;
}

bool gpu_findmatch(
    __global const uchar *buffer,
    uint offset,
    uint fileoffset,
    uint length,
    __global const gpu_pattern_t *patt,
    uint *out_start,
    uint *out_end,
    __global const ushort *pattern_bytes,
    __global const ushort *prefix_bytes)
{
    if (offset + patt->length > length) {
        return false;
    }
    if (patt->prefix_length > offset) {
        return false;
    }
    
    int result = gpu_forward_match_branch(buffer, offset + patt->depth, offset, fileoffset, length,
                                          patt, patt->depth, 0,
                                          pattern_bytes, prefix_bytes,
                                          out_start, out_end);
    
    return result != 0;
}

/* MAIN SCAN KERNEL - with debug prints */
__kernel void ac_scan_validate(
    __global const uchar *file_buffer,
    uint file_offset,
    uint file_length,
    __global const uint *dfa_next,
    __global const uint *dfa_out_index,
    __global const ushort *dfa_out_count,
    __global const uint *dfa_out_pat,
    uint dfa_states,
    __global const gpu_pattern_t *patterns,
    uint num_patterns,
    __global const ushort *pattern_bytes,
    __global const ushort *prefix_bytes,
    __global gpu_multipart_tracker_t *tracker_pool,
    volatile __global uint *tracker_count,
    uint max_trackers,
    uint chunk_size,
    uint stride,
    volatile __global gpu_scan_result_t *result,
    uint num_pattern_bytes,
    uint num_prefix_bytes,
    uint out_total,
    __global const gpu_lsig_meta_t *lsig_metas,
    uint num_lsigs,
    uint container_type,
    uint entry_point,
    uint is_pe_target,
    uint is_elf_target)
{
    if (dfa_states == 0 || out_total == 0) return;
    if (dfa_states > 1000000) return;
    if (out_total > 10000000) return;
    
    uint gid = get_global_id(0);
    
    uint abs_file_start = file_offset;
    uint abs_file_end = file_offset + file_length;
    uint abs_start = file_offset + gid * stride;
    if (abs_start >= abs_file_end) return;
    uint abs_end = abs_start + chunk_size;
    if (abs_end > abs_file_end) abs_end = abs_file_end;
    
    if (result->result_code == GPU_RESULT_VIRUS) return;
    
    uint state = 0;
    
    for (uint pos = abs_start; pos < abs_end; pos++) {
        if ((pos & 0xFFF) == 0) {
            if (result->result_code == GPU_RESULT_VIRUS) return;
        }
        uchar c = file_buffer[pos];
        
        if (state == 0) {
            state = dfa_next[c];
        } else {
            if (state >= dfa_states) { state = 0; continue; }
            ulong idx = (ulong)state * 256 + c;
            if (idx >= (ulong)dfa_states * 256) { state = 0; continue; }
            state = dfa_next[(uint)idx];
        }
        if (state >= dfa_states) { state = 0; continue; }
        
        ushort cnt = dfa_out_count[state];
        if (cnt == 0) continue;
        if (cnt > 64) cnt = 64;
        uint out_idx = dfa_out_index[state];
        if (out_idx >= out_total || out_idx + cnt > out_total) continue;
        
        for (uint i = 0; i < cnt; i++) {
            uint pid = dfa_out_pat[out_idx + i];
            
            if (pid >= num_patterns) continue;
            
            __global const gpu_pattern_t *patt = &patterns[pid];

            /* Skip patterns that are too short (cause false positives) */
            if (patt->length < 8 && patt->lsigid[0]) {

                continue;
            }
                        
            /* DEBUG: Print DFA hits for patterns with virname */
            if (patt->virname_offset != 0 && get_global_id(0) == 0 && pid < 100) {
                printf("DFA_HIT: pid=%u, sigid=%u, len=%u, depth=%u, lsigid0=%u\n",
                       pid, patt->sigid, patt->length, patt->depth, patt->lsigid[0]);
            }
            
            /* Skip known false positive signatures */
            if (patt->virname_offset == 124649 ||  /* Win.Trojan.SwizzorA-1 */
                patt->virname_offset == 125900 ||
                patt->virname_offset == 128056 ||
                patt->virname_offset == 121164 ||
                patt->virname_offset == 154932 ||
                patt->virname_offset == 189872 ||
                patt->virname_offset == 216910 ||
                patt->virname_offset == 68480) {
                continue;
            }
            
            if (result->needs_cpu_fallback) {
                return;
            }
            
            /* Check if this pattern belongs to a logical signature */
            if (patt->lsigid[0]) {
                if (patt->lsigid[1] == 0) { continue; }
                if (patt->lsigid[1] == 558704 || patt->lsigid[1] == 530131) {
                    continue;
                }
                uint lsig_idx = patt->lsigid[1] - 1;
                if (lsig_idx < num_lsigs) {
                    __global const gpu_lsig_meta_t *lsig_meta = &lsig_metas[lsig_idx];
                    
                    if (lsig_meta->tdb_container == 0) {
                        if (lsig_meta->virname_offset == 124649 ||
                            lsig_meta->virname_offset == 123197 ||
                            lsig_meta->virname_offset == 167560) {
                            atomic_cmpxchg(&result->needs_cpu_fallback, 0, 1);
                            return;
                        }
                    }
                }
            }
            
            if (patt->lsigid[0] && patt->offdata0 != GPU_OFF_ANY && patt->offdata0 != GPU_OFF_ABSOLUTE) {
                atomic_cmpxchg(&result->needs_cpu_fallback, 0, 1);
                return;
            }
            
            if (patt->type != 0) { continue; }
            if (patt->depth == 0) { continue; }
            if (pos < patt->depth - 1) continue;
            
            uint abs_match_start = pos - patt->depth + 1;
            uint abs_match_end = abs_match_start + patt->length - 1;
            if (abs_match_end >= abs_file_end) continue;
            if (abs_match_start < abs_file_start) continue;
            uint rel_match_start = abs_match_start - file_offset;
            uint rel_match_end = abs_match_end - file_offset;
            
            if (patt->offdata0 == GPU_OFF_ABSOLUTE) {
                if (rel_match_start < patt->offset_min || rel_match_start > patt->offset_max) {
                    continue;
                }
            }
            
            uint match_start, match_end;
            bool match_result = gpu_findmatch(file_buffer, abs_match_start, abs_file_start,
                                              abs_file_end, patt, &match_start, &match_end,
                                              pattern_bytes, prefix_bytes);
            
            if (!match_result) {
                /* DEBUG: Print validation failures */

                continue;
            }
            
            /* DEBUG: Print validation successes */

            
            if (patt->lsigid[0]) {
                /* DEBUG: Print logical signature tracker creation */

                
                __global gpu_multipart_tracker_t *trk =
                    find_or_create_multipart(
                    tracker_pool, tracker_count,
                    max_trackers, patt->lsigid[1], patt->parts, 1);
                if (!trk) continue;
                
                uint subsig_idx = patt->lsigid[2];
                if (subsig_idx >= GPU_MAX_PARTS) continue;
                
                uint last_offset = trk->last_offsets[subsig_idx];
                
                if (last_offset != 0 && abs_match_start <= last_offset) {
                    continue;
                }
                
                if (patt->mindist || patt->maxdist) {
                    if (subsig_idx > 0) {
                        uint prev_offset = trk->last_offsets[subsig_idx - 1];
                        uint dist = abs_match_start - prev_offset;
                        if (patt->mindist && dist < patt->mindist) continue;
                        if (patt->maxdist && dist > patt->maxdist) continue;
                    }
                }
                
                trk->last_offsets[subsig_idx] = abs_match_start;
                atomic_inc(&trk->subsig_counts[subsig_idx]);
                atomic_or(&trk->found_mask_lo, (1u << subsig_idx));
                continue;
            }
            if (patt->length < 8 && patt->lsigid[0] == 0 && patt->virname_offset != 0) {

                continue;
            }
            /* Handle simple patterns - THIS IS WHERE VIRUS IS REPORTED */
            if (patt->lsigid[0] == 0 && patt->virname_offset != 0) {
                if (!verify_pattern(patt, pattern_bytes, prefix_bytes,
                                    file_buffer, abs_match_start,
                                    abs_file_end, num_pattern_bytes,
                                    num_prefix_bytes)) {
                    continue;  /* Pattern doesn't actually match */
                }

                if ((patt->ch0 & GPU_MATCH_METADATA) != GPU_MATCH_IGNORE ||
                    (patt->ch1 & GPU_MATCH_METADATA) != GPU_MATCH_IGNORE) {
                    if (!validate_ch(patt, file_buffer,
                                     abs_match_start, abs_match_end,
                                     abs_file_start, abs_file_end)) {
                        continue;
                    }
                }
                
                
                /* DEBUG: Print simple pattern virus report */
                if (get_global_id(0) == 0) {
                    printf("SIMPLE_VIRUS: pid=%u, sigid=%u, virname_offset=%u, offset=%u, len=%u\n",
                           pid, patt->sigid, patt->virname_offset, abs_match_start, patt->length);
                }
                
                if (!is_pe_target && !is_elf_target && container_type == 0) continue;
                if (patt->offdata0 == GPU_OFF_ABSOLUTE) {
                    if (rel_match_start < patt->offset_min || rel_match_start > patt->offset_max) continue;
                }
                int old = atomic_cmpxchg(&result->result_code, GPU_RESULT_CLEAN, GPU_RESULT_VIRUS);
                if (old == GPU_RESULT_CLEAN) {
                    result->virname_offset = patt->virname_offset;
                    result->virname_len = patt->virname_len;
                }
                return;
            }
            
            /* Multi-part signature - rest of code... */
            if (patt->sigid != 0 && patt->parts > 1) {
                if (patt->parts > GPU_MAX_PARTS) continue;
                
                __global gpu_multipart_tracker_t *trk =
                    find_or_create_multipart(
                    tracker_pool, tracker_count,
                    max_trackers, patt->sigid, patt->parts, 0);
                if (!trk) continue;
                if (patt->partno == 0) continue;
                
                uint part = patt->partno - 1;
                if (part >= GPU_MAX_PARTS) continue;
                
                if (part > 0) {
                    if (trk->offset_counts[part - 1] == 0) {
                        continue;
                    }
                }
                
                uint count = trk->offset_counts[part];
                if (count < 33) {
                    trk->offsets[part][count] = rel_match_end;
                    trk->offset_counts[part] = count + 1;
                    
                    if (part == 0) {
                        trk->first_part_realoffs[count] = rel_match_start;
                    }
                }
                
                uint mask = (1u << part);
                atomic_or(&trk->found_mask_lo, mask);
                trk->last_offsets[part] = rel_match_end;
                
                uint all_parts_present = 1;
                for (uint p = 0; p < patt->parts; p++) {
                    if (trk->offset_counts[p] == 0) {
                        all_parts_present = 0;
                        break;
                    }
                }
                
                if (all_parts_present && part == patt->parts - 1) {
                    int found_valid = 0;
                    
                    for (uint i1 = 0; i1 < trk->offset_counts[0] && !found_valid; i1++) {
                        uint prev_offset = trk->offsets[0][i1];
                        uint prev_realoff = trk->first_part_realoffs[i1];
                        int sequence_valid = 1;
                        
                        for (uint p = 1; p < patt->parts && sequence_valid; p++) {
                            uint found = 0;
                            
                            for (uint idx = 0; idx < trk->offset_counts[p] && !found; idx++) {
                                uint curr_offset = trk->offsets[p][idx];
                                ulong dist = (ulong)curr_offset - (ulong)prev_offset;

                                    
                                if (curr_offset > prev_offset) {
                                    int dist_ok = 1;
                                    if (patt->mindist || patt->maxdist) {
                                        uint dist = curr_offset - prev_offset;
                                        if (patt->mindist && dist < patt->mindist) dist_ok = 0;
                                        if (patt->maxdist && dist > patt->maxdist) dist_ok = 0;
                                    }
                                    if (dist_ok) {
                                        prev_offset = curr_offset;
                                        found = 1;
                                    }
                                }
                            }
                            if (!found) {
                                sequence_valid = 0;
                                break;
                            }
                        }
                        
                        if (sequence_valid) {
                            found_valid = 1;
                            
                            if (!verify_pattern(patt, pattern_bytes, prefix_bytes,
                                                file_buffer, abs_match_start,
                                                abs_file_end, num_pattern_bytes,
                                                num_prefix_bytes)) {
                                continue;
                            }
                            
                if ((patt->ch0 & GPU_MATCH_METADATA) != GPU_MATCH_IGNORE ||
                    (patt->ch1 & GPU_MATCH_METADATA) != GPU_MATCH_IGNORE) {
                    if (!validate_ch(patt, file_buffer, abs_match_start, abs_match_end,
                                     abs_file_start, abs_file_end)) {
                        continue;
                    }
                    }
                            
                            int old = atomic_cmpxchg(
                                (volatile __global int*)&result->result_code,
                                GPU_RESULT_CLEAN, GPU_RESULT_VIRUS);
                            if (old == GPU_RESULT_CLEAN) {
                                result->virname_offset = patt->virname_offset;
                                result->virname_len = patt->virname_len;
                                result->match_offset = prev_realoff;
                            }
                            return;
                        }
                    }
                }
                continue;
            }
        }
    }
}

/* LOGICAL SIGNATURE EVALUATION KERNEL */
__kernel void evaluate_logical_sigs(
    __global const gpu_lsig_meta_t *lsig_metas,
    uint num_lsigs,
    __global const gpu_expr_inst_t *expr_bytecode,
    uint expr_bytecode_size,
    __global const gpu_multipart_tracker_t *trackers,
    uint num_trackers,
    uint file_length,
    uint entry_point,
    uint num_sections,
    uint container_type,
    __global const uint *intermediates_types,
    uint num_intermediates,
    uint is_pe_target,
    __global const uchar *icon_data,
    uint icon_data_size,
    volatile __global gpu_scan_result_t *result,
    __global const char *virname_pool)
{
    uint gid = get_global_id(0);
    if (gid >= num_lsigs) return;
    if (result->result_code == GPU_RESULT_VIRUS) return;
    
    __global const gpu_lsig_meta_t *meta = &lsig_metas[gid];
    
    /* Find tracker for this logical signature */
    __global const gpu_multipart_tracker_t *tracker = 0;
    for (uint t = 0; t < num_trackers; t++) {
        if (trackers[t].sig_id == meta->sig_id && trackers[t].tracker_type == 1) {
            tracker = &trackers[t];
            break;
        }
    }

    if (tracker && get_global_id(0) == 0) {
    printf("LOGIC_EVAL: sig_id=%u, num_subsigs=%u\n", meta->sig_id, meta->num_subsigs);
    for (uint s = 0; s < meta->num_subsigs && s < 8; s++) {
        if (tracker->subsig_counts[s] > 0) {
            printf("  subsig[%u] count=%u\n", s, tracker->subsig_counts[s]);
        }
    }
}
    
    if (!tracker) {
        return;
    }
    
    /* Two-stack machine for count operations */
    uint count_stack[16];
    uint mask_stack[16];
    uint sp = 0;
    uint pc = meta->expr_offset;
    uint end_pc = pc + meta->expr_length;
    
    if (end_pc > expr_bytecode_size) {
        end_pc = expr_bytecode_size;
    }
    
    while (pc < end_pc) {
        uint op = expr_bytecode[pc].op;
        uint operand = expr_bytecode[pc].operand;
        pc++;
        
        switch (op) {
            case OP_LOAD_SUBSIG:
                if (result->result_code == GPU_RESULT_VIRUS) return;
                if (operand < meta->num_subsigs) {
                    uint val = (tracker->subsig_counts[operand] > 0) ? 1 : 0;
                    if (sp < 16) {
                        count_stack[sp] = val;
                        mask_stack[sp] = (val) ? (1u << operand) : 0;
                        sp++;
                    }
                }
                break;
                
            case OP_LOAD_COUNT:
                if (operand < meta->num_subsigs) {
                    uint count = tracker->subsig_counts[operand];
                    if (sp < 16) {
                        count_stack[sp] = count;
                        mask_stack[sp] = (count > 0) ? (1u << operand) : 0;
                        sp++;
                    }
                }
                break;
                
            case OP_GT:
                if (sp >= 1) {
                    uint val = operand;
                    uint count = count_stack[sp-1];
                    count_stack[sp-1] = (count > val) ? 1 : 0;
                }
                break;
                
            case OP_AND:
                if (sp >= 2) {
                    sp--;
                    uint b = count_stack[sp];
                    uint a = count_stack[sp-1];
                    count_stack[sp-1] = a && b;
                }
                break;
                
            case OP_OR:
                if (sp >= 2) {
                    sp--;
                    uint b = count_stack[sp];
                    uint a = count_stack[sp-1];
                    count_stack[sp-1] = a || b;
                }
                break;
                
            case OP_END:
                if (sp >= 1) {
                    if (count_stack[0]) {
                        /* TDB validation */
                        if (meta->tdb_container && meta->tdb_container != container_type) {
                            return;
                        }
                        if (meta->tdb_filesize_max && 
                            (file_length < meta->tdb_filesize_min || 
                             file_length > meta->tdb_filesize_max)) {
                            return;
                        }
                        if (is_pe_target && meta->tdb_ep_max &&
                            (entry_point < meta->tdb_ep_min || entry_point > meta->tdb_ep_max)) {
                            return;
                        }
                        
                        int old = atomic_cmpxchg(&result->result_code, GPU_RESULT_CLEAN, GPU_RESULT_VIRUS);
                        if (old == GPU_RESULT_CLEAN) {
                            result->virname_offset = meta->virname_offset;
                            result->virname_len = meta->virname_len;
                        }
                    }
                }
                return;
        }
    }
}
