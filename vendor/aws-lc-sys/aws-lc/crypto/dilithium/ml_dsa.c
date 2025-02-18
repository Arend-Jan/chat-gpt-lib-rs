// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0 OR ISC

#include "../evp_extra/internal.h"
#include "../fipsmodule/evp/internal.h"
#include "ml_dsa.h"
#include "pqcrystals_dilithium_ref_common/sign.h"
#include "pqcrystals_dilithium_ref_common/params.h"

// These includes are required to compile ML-DSA. These can be moved to bcm.c
// when ML-DSA is added to the fipsmodule directory.
#include "./pqcrystals_dilithium_ref_common/ntt.c"
#include "./pqcrystals_dilithium_ref_common/packing.c"
#include "./pqcrystals_dilithium_ref_common/params.c"
#include "./pqcrystals_dilithium_ref_common/poly.c"
#include "./pqcrystals_dilithium_ref_common/polyvec.c"
#include "./pqcrystals_dilithium_ref_common/reduce.c"
#include "./pqcrystals_dilithium_ref_common/rounding.c"
#include "./pqcrystals_dilithium_ref_common/sign.c"

// Note: These methods currently default to using the reference code for
// ML-DSA. In a future where AWS-LC has optimized options available,
// those can be conditionally (or based on compile-time flags) called here,
// depending on platform support.

int ml_dsa_44_keypair(uint8_t *public_key   /* OUT */,
                      uint8_t *private_key  /* OUT */) {
  ml_dsa_params params;
  ml_dsa_44_params_init(&params);
  return (ml_dsa_keypair(&params, public_key, private_key) == 0);
}

int ml_dsa_44_keypair_internal(uint8_t *public_key   /* OUT */,
                               uint8_t *private_key  /* OUT */,
                               const uint8_t *seed   /* IN */) {
  ml_dsa_params params;
  ml_dsa_44_params_init(&params);
  return ml_dsa_keypair_internal(&params, public_key, private_key, seed) == 0;
}

int ml_dsa_44_sign(const uint8_t *private_key /* IN */,
                   uint8_t *sig               /* OUT */,
                   size_t *sig_len            /* OUT */,
                   const uint8_t *message     /* IN */,
                   size_t message_len         /* IN */,
                   const uint8_t *ctx_string  /* IN */,
                   size_t ctx_string_len      /* IN */) {
  ml_dsa_params params;
  ml_dsa_44_params_init(&params);
  return ml_dsa_sign(&params, sig, sig_len, message, message_len,
                     ctx_string, ctx_string_len, private_key) == 0;
}

int ml_dsa_44_sign_internal(const uint8_t *private_key  /* IN */,
                            uint8_t *sig                /* OUT */,
                            size_t *sig_len             /* OUT */,
                            const uint8_t *message      /* IN */,
                            size_t message_len          /* IN */,
                            const uint8_t *pre          /* IN */,
                            size_t pre_len              /* IN */,
                            uint8_t *rnd                /* IN */) {
  ml_dsa_params params;
  ml_dsa_44_params_init(&params);
  return ml_dsa_sign_internal(&params, sig, sig_len, message, message_len,
                              pre, pre_len, rnd, private_key) == 0;
}

int ml_dsa_44_verify(const uint8_t *public_key /* IN */,
                     const uint8_t *sig        /* IN */,
                     size_t sig_len            /* IN */,
                     const uint8_t *message    /* IN */,
                     size_t message_len        /* IN */,
                     const uint8_t *ctx_string /* IN */,
                     size_t ctx_string_len     /* IN */) {
  ml_dsa_params params;
  ml_dsa_44_params_init(&params);
  return ml_dsa_verify(&params, sig, sig_len, message, message_len,
                       ctx_string, ctx_string_len, public_key) == 0;
}

int ml_dsa_44_verify_internal(const uint8_t *public_key /* IN */,
                              const uint8_t *sig        /* IN */,
                              size_t sig_len            /* IN */,
                              const uint8_t *message    /* IN */,
                              size_t message_len        /* IN */,
                              const uint8_t *pre        /* IN */,
                              size_t pre_len            /* IN */) {
  ml_dsa_params params;
  ml_dsa_44_params_init(&params);
  return ml_dsa_verify_internal(&params, sig, sig_len, message, message_len,
                                pre, pre_len, public_key) == 0;
}

int ml_dsa_65_keypair(uint8_t *public_key   /* OUT */,
                      uint8_t *private_key  /* OUT */) {
  ml_dsa_params params;
  ml_dsa_65_params_init(&params);
  return (ml_dsa_keypair(&params, public_key, private_key) == 0);
}

int ml_dsa_65_keypair_internal(uint8_t *public_key   /* OUT */,
                               uint8_t *private_key  /* OUT */,
                               const uint8_t *seed   /* IN */) {
  ml_dsa_params params;
  ml_dsa_65_params_init(&params);
  return ml_dsa_keypair_internal(&params, public_key, private_key, seed) == 0;
}

int ml_dsa_65_sign(const uint8_t *private_key /* IN */,
                   uint8_t *sig               /* OUT */,
                   size_t *sig_len            /* OUT */,
                   const uint8_t *message     /* IN */,
                   size_t message_len         /* IN */,
                   const uint8_t *ctx_string  /* IN */,
                   size_t ctx_string_len      /* IN */) {
  ml_dsa_params params;
  ml_dsa_65_params_init(&params);
  return ml_dsa_sign(&params, sig, sig_len, message, message_len,
                     ctx_string, ctx_string_len, private_key) == 0;
}

int ml_dsa_65_sign_internal(const uint8_t *private_key  /* IN */,
                            uint8_t *sig                /* OUT */,
                            size_t *sig_len             /* OUT */,
                            const uint8_t *message      /* IN */,
                            size_t message_len          /* IN */,
                            const uint8_t *pre          /* IN */,
                            size_t pre_len              /* IN */,
                            uint8_t *rnd                /* IN */) {
  ml_dsa_params params;
  ml_dsa_65_params_init(&params);
  return ml_dsa_sign_internal(&params, sig, sig_len, message, message_len,
                              pre, pre_len, rnd, private_key) == 0;
}

int ml_dsa_65_verify(const uint8_t *public_key /* IN */,
                     const uint8_t *sig        /* IN */,
                     size_t sig_len            /* IN */,
                     const uint8_t *message    /* IN */,
                     size_t message_len        /* IN */,
                     const uint8_t *ctx_string /* IN */,
                     size_t ctx_string_len     /* IN */) {
  ml_dsa_params params;
  ml_dsa_65_params_init(&params);
  return ml_dsa_verify(&params, sig, sig_len, message, message_len,
                       ctx_string, ctx_string_len, public_key) == 0;
}

int ml_dsa_65_verify_internal(const uint8_t *public_key /* IN */,
                              const uint8_t *sig        /* IN */,
                              size_t sig_len            /* IN */,
                              const uint8_t *message    /* IN */,
                              size_t message_len        /* IN */,
                              const uint8_t *pre        /* IN */,
                              size_t pre_len            /* IN */) {
  ml_dsa_params params;
  ml_dsa_65_params_init(&params);
  return ml_dsa_verify_internal(&params, sig, sig_len, message, message_len,
                                pre, pre_len, public_key) == 0;
}

int ml_dsa_87_keypair(uint8_t *public_key   /* OUT */,
                      uint8_t *private_key  /* OUT */) {
  ml_dsa_params params;
  ml_dsa_87_params_init(&params);
  return (ml_dsa_keypair(&params, public_key, private_key) == 0);
}

int ml_dsa_87_keypair_internal(uint8_t *public_key   /* OUT */,
                               uint8_t *private_key  /* OUT */,
                               const uint8_t *seed   /* IN */) {
  ml_dsa_params params;
  ml_dsa_87_params_init(&params);
  return ml_dsa_keypair_internal(&params, public_key, private_key, seed) == 0;
}

int ml_dsa_87_sign(const uint8_t *private_key /* IN */,
                   uint8_t *sig               /* OUT */,
                   size_t *sig_len            /* OUT */,
                   const uint8_t *message     /* IN */,
                   size_t message_len         /* IN */,
                   const uint8_t *ctx_string  /* IN */,
                   size_t ctx_string_len      /* IN */) {
  ml_dsa_params params;
  ml_dsa_87_params_init(&params);
  return ml_dsa_sign(&params, sig, sig_len, message, message_len,
                     ctx_string, ctx_string_len, private_key) == 0;
}

int ml_dsa_87_sign_internal(const uint8_t *private_key  /* IN */,
                            uint8_t *sig                /* OUT */,
                            size_t *sig_len             /* OUT */,
                            const uint8_t *message      /* IN */,
                            size_t message_len          /* IN */,
                            const uint8_t *pre          /* IN */,
                            size_t pre_len              /* IN */,
                            uint8_t *rnd                /* IN */) {
  ml_dsa_params params;
  ml_dsa_87_params_init(&params);
  return ml_dsa_sign_internal(&params, sig, sig_len, message, message_len,
                              pre, pre_len, rnd, private_key) == 0;
}

int ml_dsa_87_verify(const uint8_t *public_key /* IN */,
                     const uint8_t *sig        /* IN */,
                     size_t sig_len            /* IN */,
                     const uint8_t *message    /* IN */,
                     size_t message_len        /* IN */,
                     const uint8_t *ctx_string /* IN */,
                     size_t ctx_string_len     /* IN */) {
  ml_dsa_params params;
  ml_dsa_87_params_init(&params);
  return ml_dsa_verify(&params, sig, sig_len, message, message_len,
                       ctx_string, ctx_string_len, public_key) == 0;
}

int ml_dsa_87_verify_internal(const uint8_t *public_key /* IN */,
                              const uint8_t *sig        /* IN */,
                              size_t sig_len            /* IN */,
                              const uint8_t *message    /* IN */,
                              size_t message_len        /* IN */,
                              const uint8_t *pre        /* IN */,
                              size_t pre_len            /* IN */) {
  ml_dsa_params params;
  ml_dsa_87_params_init(&params);
  return ml_dsa_verify_internal(&params, sig, sig_len, message, message_len,
                                pre, pre_len, public_key) == 0;
}
