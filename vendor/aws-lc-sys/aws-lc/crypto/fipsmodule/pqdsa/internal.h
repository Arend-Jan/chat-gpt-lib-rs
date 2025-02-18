// Copyright Amazon.com Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0 OR ISC

#ifndef AWSLC_HEADER_PQDSA_INTERNAL_H
#define AWSLC_HEADER_PQDSA_INTERNAL_H

#include <openssl/base.h>

#if defined(__cplusplus)
extern "C" {
#endif

// PQDSA_METHOD structure and helper functions.
typedef struct {
  int (*pqdsa_keygen)(uint8_t *public_key,
                      uint8_t *private_key);

  int (*pqdsa_sign)(const uint8_t *private_key,
                    uint8_t *sig,
                    size_t *sig_len,
                    const uint8_t *message,
                    size_t message_len,
                    const uint8_t *ctx_string,
                    size_t ctx_string_len);

  int (*pqdsa_verify)(const uint8_t *public_key,
                      const uint8_t *sig,
                      size_t sig_len,
                      const uint8_t *message,
                      size_t message_len,
                      const uint8_t *ctx_string,
                      size_t ctx_string_len);

} PQDSA_METHOD;

// PQDSA structure and helper functions.
typedef struct {
  int nid;
  const uint8_t *oid;
  uint8_t oid_len;
  const char *comment;
  size_t public_key_len;
  size_t private_key_len;
  size_t signature_len;
  size_t keygen_seed_len;
  size_t sign_seed_len;
  const PQDSA_METHOD *method;
} PQDSA;

// PQDSA_KEY structure and helper functions.
struct pqdsa_key_st {
  const PQDSA *pqdsa;
  uint8_t *public_key;
  uint8_t *private_key;
};

int PQDSA_KEY_init(PQDSA_KEY *key, const PQDSA *pqdsa);
const PQDSA * PQDSA_find_dsa_by_nid(int nid);
const EVP_PKEY_ASN1_METHOD *PQDSA_find_asn1_by_nid(int nid);
const PQDSA *PQDSA_KEY_get0_dsa(PQDSA_KEY* key);
PQDSA_KEY *PQDSA_KEY_new(void);
void PQDSA_KEY_free(PQDSA_KEY *key);
int EVP_PKEY_pqdsa_set_params(EVP_PKEY *pkey, int nid);

int PQDSA_KEY_set_raw_public_key(PQDSA_KEY *key, const uint8_t *in);
int PQDSA_KEY_set_raw_private_key(PQDSA_KEY *key, const uint8_t *in);
#if defined(__cplusplus)
}  // extern C
#endif

#endif // AWSLC_HEADER_DSA_TEST_INTERNAL_H
