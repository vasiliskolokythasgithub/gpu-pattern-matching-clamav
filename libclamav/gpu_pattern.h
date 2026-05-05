/*
 * gpu_pattern_t - Pattern metadata for GPU on-device validation
 *
 * One struct per pattern, uploaded once at engine init.
 * Layout MUST match the OpenCL kernel typedef exactly.
 *
 * Add to gpu_ac.h
 */

#ifndef GPU_PATTERN_H
#define GPU_PATTERN_H

#include <stdint.h>

/* These must match the kernel defines */
#define GPU_MATCH_CHAR        0x0000
#define GPU_MATCH_IGNORE      0x0200
#define GPU_MATCH_NOCASE      0x0400
#define GPU_MATCH_NIBBLE_HIGH 0x0100
#define GPU_MATCH_NIBBLE_LOW  0x0300
#define GPU_MATCH_METADATA    0xFF00

#define GPU_OFF_ANY      0
#define GPU_OFF_ABSOLUTE 1
#define GPU_OFF_NONE     0xFFFFFFFF

#define GPU_RESULT_CLEAN  0
#define GPU_RESULT_VIRUS  1
#define GPU_RESULT_BREAK  2

#define GPU_MAX_PARTS    16
#define GPU_MAX_TRACKERS 65536

/*
 * Pattern metadata - packed to match OpenCL struct
 * Total size: 64 bytes (no padding issues)
 */
typedef struct {
    uint32_t length;
    uint32_t prefix_length;
    uint32_t parts;
    uint32_t type;
    uint32_t sigid;
    uint32_t partno;
    uint32_t offset_min;
    uint32_t offset_max;
    uint32_t offdata0;
    uint32_t lsigid[3];  
    uint32_t ch0;
    uint32_t ch1;
    uint32_t ch_mindist0;
    uint32_t ch_mindist1;
    uint32_t ch_maxdist0;
    uint32_t ch_maxdist1;
    uint32_t mindist;
    uint32_t maxdist;
    uint32_t pattern_offset;
    uint32_t prefix_offset;
    uint32_t virname_offset;
    uint32_t virname_len;
    uint32_t depth;
    uint32_t boundary;        /* ADD THIS - boundary flags */
    uint32_t sigopts;
    uint32_t is_bytecode;    
    uint32_t has_regex; 
    uint32_t   special_pattern;
} gpu_pattern_t;  /* 92 bytes */





/*
 * Scan result - read back from GPU (16 bytes)
 */
typedef struct {
    int32_t  result_code;
    uint32_t virname_offset;
    uint32_t virname_len;
    uint32_t match_offset;
    uint32_t needs_cpu_fallback;  
    uint32_t fallback_offset;      
    uint32_t fallback_pattern_id;  
    uint32_t sig_id;  /* Track which signature matched for priority */
} gpu_scan_result_t;  

#endif /* GPU_PATTERN_H */


