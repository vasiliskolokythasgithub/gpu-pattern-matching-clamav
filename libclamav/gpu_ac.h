/* libclamav/gpu_ac.h */
#ifndef GPU_AC_H
#define GPU_AC_H
#include "gpu_pattern.h"
#include "clamav.h"
#include "matcher.h"
#include "matcher-ac.h"
#include <stdint.h>
#include <stdbool.h>
#define GPU_MAX_LSIG_SUBSIGS 64
#define GPU_MAX_EXPR_NODES 256

#ifndef GPU_MAX_EXPR_NODES
#define GPU_MAX_EXPR_NODES 1024  /* Add this! */
#endif

#ifdef HAVE_OPENCL
#include <CL/cl.h>
#define GPU_QUEUE_DEPTH 4  /* Number of in-flight GPU jobs */

struct gpu_job {
    cl_mem buf;                  /* GPU buffer with file data */
    bool buf_is_temp;            /* Whether to free after */
    uint32_t length;             /* File size */
    const unsigned char *host_buf; /* Host pointer for CPU validation */
    struct cli_matcher *root;    /* Matcher for this job */
    cli_ctx *ctx;                /* Scan context */
    const char **virname;        /* Where to store result */
    cl_event kernel_done;        /* Event: kernel finished */
    cl_event count_done;         /* Event: hit_count read done */
    uint32_t hit_count;          /* Result: number of hits */
    
    /* Double-buffered hit data */
    cl_mem hit_data_buf;         /* Per-job GPU hit buffer */
    cl_mem hit_count_buf;        /* Per-job hit counter */
    cl_mem pat_seen_buf;         /* Per-job pat_seen */
    cl_mem stop_flag_buf;        /* Per-job stop flag */
    
    enum { JOB_EMPTY, JOB_SUBMITTED, JOB_READY } state;
};

struct gpu_pipeline {
    struct gpu_job jobs[GPU_QUEUE_DEPTH];
    uint32_t submit_idx;    /* Next slot to submit into */
    uint32_t retire_idx;    /* Next slot to retire/collect */
    uint32_t active_count;  /* Jobs in flight */
};

typedef struct gpu_multipart_tracker {
   uint32_t   sig_id;
   uint32_t   tracker_type;
   uint32_t   total_parts;
   uint32_t   lookup_idx;  
   uint32_t   found_mask_lo;
   uint32_t   found_mask_hi;
   uint32_t   offsets[GPU_MAX_PARTS][33];
   uint32_t   offset_counts[GPU_MAX_PARTS];
   uint32_t   first_part_realoffs[33];
   uint32_t   last_offsets[GPU_MAX_PARTS];
   uint32_t   subsig_counts[GPU_MAX_PARTS];
   uint32_t   subsig_first_offset[GPU_MAX_PARTS];
   uint32_t   subsig_last_offset[GPU_MAX_PARTS];
} gpu_multipart_tracker_t;
 
typedef struct gpu_lsig_meta{
    uint32_t sig_id;
    uint32_t num_subsigs;
    uint32_t expr_offset;
    uint32_t expr_length;
    uint32_t virname_offset;
    uint32_t virname_len;
    uint32_t tdb_container;
    uint32_t tdb_filesize_min;
    uint32_t tdb_filesize_max;
    uint32_t tdb_ep_min;
    uint32_t tdb_ep_max;
    uint32_t tdb_nos_min;
    uint32_t tdb_nos_max;
    uint32_t tdb_intermediates_mask;
    uint32_t tdb_handlertype;
    uint32_t tdb_icongrp1_offset;
    uint32_t tdb_icongrp2_offset;
    uint32_t bc_idx;
    uint32_t has_regex; 
} gpu_lsig_meta_t;
 

 
/* ============ LEGACY DFA STRUCTURE ============ */
struct gpu_ac_dfa {
    uint32_t states;
    uint32_t pat_count;       // number of patterns
    uint8_t *is_simple;       // per-pattern simple flag
    uint32_t *next;
    uint32_t *out_index;
    uint16_t *out_count;
    uint16_t *out_pat;
    uint32_t out_total;
};

/* ============ FLATTENED DFA STRUCTURES ============ */

/* Flattened DFA for GPU scanning */
struct gpu_flattened_dfa {
    uint32_t states;           /* Number of DFA states */
    uint32_t *next;           /* Transition table [states * 256] */
    uint32_t *out_index;      /* Output index per state */
    uint16_t *out_count;      /* Output count per state */
    uint16_t *out_pat;        /* Pattern IDs */
    uint32_t *sig_id;         /* Signature ID (0 = simple/single-part) */
    uint32_t *part_no;        /* Part number for multi-part signatures */
    uint32_t pat_count;       /* Total patterns */
    uint32_t out_total;       /* Total output entries */
};

/* GPU scan results with flattened signature metadata */
struct gpu_ac_hits {
    uint32_t count;
    uint32_t *pat_ids;
    uint32_t *offsets;
    uint32_t *sig_ids;        /* Signature IDs for hits */
    uint32_t *part_nos;       /* Part numbers for hits */
};

/* Multi-part signature tracker for CPU validation */
struct gpu_multi_part_tracker {
    uint32_t sig_id;          /* Signature ID */
    uint32_t total_parts;     /* Total parts in signature */
    uint32_t parts_found;     /* Bitmask of parts found */
    uint32_t *offsets;        /* Offsets for each part */
    uint32_t *part_nos;       /* Part numbers */
    const char *virname;      /* Virus name if all parts found */
};


struct gpu_multi_tracker {
    uint32_t sig_id;
    uint32_t total_parts;
    uint64_t parts_found;
    uint32_t *offsets;
    uint32_t *part_nos;
    const char *virname;
    struct cli_ac_patt **patterns;
};

cl_error_t gpu_async_submit(struct gpu_rt *rt, const unsigned char *buffer,
                             uint32_t length, struct cli_matcher *root);
cl_error_t gpu_async_collect(struct gpu_rt *rt, const unsigned char *buffer,
                              uint32_t length, const char **virname,
                              struct cli_matcher *root, cli_ctx *ctx);

static cl_error_t gpu_validate_hits(uint32_t *hit_data, uint32_t hit_count,
                                     const unsigned char *buffer, uint32_t length,
                                     const char **virname,
                                     struct cli_matcher *root, cli_ctx *ctx);


cl_error_t gpu_scan_single(struct gpu_rt *rt,
                            const unsigned char *buffer,
                            uint32_t length,
                            struct cli_matcher *root,
                            uint8_t **chunk_bitmap_out,
                            uint32_t *num_chunks_out);


 

/* Expression bytecode instructions */
typedef enum {
    OP_LOAD_SUBSIG = 0,  /* Push subsig match status (operand = subsig index) */
    OP_AND = 1,           /* Pop two, push AND */
    OP_OR = 2,            /* Pop two, push OR */
    OP_NOT = 3,           /* Pop one, push NOT */
    OP_END = 4            /* End of expression, result on stack */
} expr_op_t;

typedef struct {
    uint32_t op;
    uint32_t operand;      /* For OP_LOAD: subsig index */
} gpu_expr_inst_t;

/* ============ GPU RUNTIME STRUCTURE ============ */

 struct gpu_rt {
    /* OpenCL objects */
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel; 
    cl_kernel lsig_kernel;      /* Only declare once */
    uint32_t max_trackers; 
 

    uint32_t icon_data_capacity;    
    cl_mem d_intermediates;    /* Add this */
    cl_mem d_icon_data;        /* Add this */
 
        /* NEW - persistent buffer for icon data */
    uint32_t intermediates_capacity; /* NEW - size of intermediates buffer
    
    /* Pattern metadata buffers */
    cl_mem d_patterns;
    cl_mem d_pattern_bytes;
    cl_mem d_prefix_bytes;
    cl_mem d_virname_pool;
    cl_mem d_tracker_pool;
    cl_mem d_tracker_count;
    cl_mem d_result;
    cl_mem d_lsig_metas;         /* Only once */
    cl_mem d_expr_bytecode;      /* Only once */
    
    
    /* DFA buffers (GPU side) */
    cl_mem dfa_next;
    cl_mem dfa_out_index;
    cl_mem dfa_out_count;
    cl_mem dfa_out_pat;
    cl_mem dfa_sig_id;
    cl_mem dfa_part_no;
    
    /* Working buffers */
    cl_mem hit_count;
    cl_mem hit_data;
    cl_mem stop_flag;
    cl_mem pat_seen;
    cl_mem scan_buffer;
    cl_mem is_simple;
    
    /* Host copy of virname pool */
    char *h_virname_pool;
    
    /* Buffer sizes */
    uint32_t dfa_states;
    uint32_t patt_count;
    uint32_t pat_seen_count;
    uint32_t num_pattern_bytes;
    uint32_t num_prefix_bytes;
    uint32_t out_total;
    uint32_t num_gpu_patterns;
    uint32_t num_lsigs;           /* Only once */
    uint32_t expr_bytecode_size;
    uint32_t virname_pool_size;
    size_t scan_buffer_size;
    
    /* State */
    int initialized;
    int dfa_uploaded;
    int v2_uploaded;
    
    /* Async state */
    bool async_submitted;
    cl_mem async_buf;
    bool async_buf_is_temp;
    uint32_t async_length;
    uint32_t async_hit_count;
    cl_event async_kernel_event;
    cl_event async_count_event;
    
    /* Batch processing support */
    struct {
        const unsigned char **buffers;
        uint32_t *lengths;
        uint32_t *file_offsets;
        uint32_t count;
        uint32_t max_count;
        size_t total_bytes;
        cl_error_t *results;
        const char **virnames;
        cl_mem batch_buffer;
        size_t batch_buffer_size;
        cl_event write_event;
        cl_event *kernel_events;
        cl_event *count_events;
        uint32_t *hit_counts;
        bool active;
        uint32_t files_processed;
    } batch;
};

 
/* Function prototypes */
cl_error_t gpu_process_batch(struct gpu_rt *rt, struct cli_matcher *root);
cl_error_t gpu_collect_batch_results(struct gpu_rt *rt, struct cli_matcher *root,
                                      cli_ctx *ctx, const char **virname);
 
/* ============ FUNCTION PROTOTYPES ============ */

cl_error_t gpu_cache_lookup(struct scan_cache *cache, const unsigned char *buf, uint32_t len, cl_error_t *result);
void gpu_cache_insert(struct scan_cache *cache, const unsigned char *buf, uint32_t len, cl_error_t result);

/* Flattened DFA functions */
struct gpu_flattened_dfa *gpu_build_flattened_dfa(struct cli_matcher *root);
int gpu_rt_upload_flattened_dfa(struct gpu_rt *rt, struct gpu_flattened_dfa *dfa);
cl_error_t cli_ac_scanbuff_gpu_flattened(
    const unsigned char *buffer,
    uint32_t length,
    const char **virname,
    struct cli_matcher *root,
    cli_ctx *ctx
);

struct gpu_hint {
    uint32_t pat_id;
    uint32_t offset;
    uint32_t sig_id;
    uint32_t part_no;
};





/* Legacy DFA functions (backward compatibility) */
int gpu_rt_init(struct gpu_rt *rt);
void gpu_rt_destroy(struct gpu_rt *rt);
int gpu_rt_upload_dfa(struct gpu_rt *rt, const struct gpu_ac_dfa *dfa);
cl_error_t cli_ac_scanbuff_gpu_dfa(
    const unsigned char *buffer,
    uint32_t length,
    const char **virname,
    struct cli_ac_result **res,
    const struct cli_matcher *root,
    struct cli_ac_data *mdata,
    uint32_t base_offset,
    unsigned int mode,
    cli_ctx *ctx
);
void gpu_extract_dfa(struct cli_matcher *root);

/* Helper functions */
static inline int gpu_ac_should_run(const struct cli_matcher *root,
                                    uint32_t length)
{
    if (length < 16 * 1024)
        return 0;

    if (root->gpu_dfa_states == 0 || root->gpu_patt_count == 0)
        return 0;

    uint64_t work = (uint64_t)root->gpu_dfa_states * length;
    return work >= (8ULL * 1024 * 1024);
}

#endif /* HAVE_OPENCL */
#endif /* GPU_AC_H */