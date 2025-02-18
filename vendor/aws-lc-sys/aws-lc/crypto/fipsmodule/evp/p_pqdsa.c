// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0 OR ISC

#include <openssl/evp.h>
#include <openssl/err.h>
#include <openssl/mem.h>

#include "../delocate.h"
#include "../crypto/dilithium/ml_dsa.h"
#include "../crypto/evp_extra/internal.h"
#include "../crypto/internal.h"
#include "../pqdsa/internal.h"

// PQDSA PKEY functions

typedef struct {
  const PQDSA *pqdsa;
} PQDSA_PKEY_CTX;

static int pkey_pqdsa_init(EVP_PKEY_CTX *ctx) {
  PQDSA_PKEY_CTX *dctx;
  dctx = OPENSSL_zalloc(sizeof(PQDSA_PKEY_CTX));
  if (dctx == NULL) {
    return 0;
  }

  ctx->data = dctx;

  return 1;
}

static void pkey_pqdsa_cleanup(EVP_PKEY_CTX *ctx) {
  OPENSSL_free(ctx->data);
}

static int pkey_pqdsa_keygen(EVP_PKEY_CTX *ctx, EVP_PKEY *pkey) {
  GUARD_PTR(ctx);
  PQDSA_PKEY_CTX *dctx = ctx->data;
  GUARD_PTR(dctx);
  const PQDSA *pqdsa = dctx->pqdsa;
  if (pqdsa == NULL) {
    if (ctx->pkey == NULL) {
      OPENSSL_PUT_ERROR(EVP, EVP_R_NO_PARAMETERS_SET);
      return 0;
    }
    pqdsa = PQDSA_KEY_get0_dsa(ctx->pkey->pkey.pqdsa_key);
  }

  PQDSA_KEY *key = PQDSA_KEY_new();
  if (key == NULL ||
      !PQDSA_KEY_init(key, pqdsa) ||
      !pqdsa->method->pqdsa_keygen(key->public_key, key->private_key) ||
      !EVP_PKEY_assign(pkey, EVP_PKEY_PQDSA, key)) {
    PQDSA_KEY_free(key);
    return 0;
      }
  return 1;
}

static int pkey_pqdsa_sign_message(EVP_PKEY_CTX *ctx, uint8_t *sig,
                                     size_t *sig_len, const uint8_t *message,
                                     size_t message_len) {
  PQDSA_PKEY_CTX *dctx = ctx->data;
  const PQDSA *pqdsa = dctx->pqdsa;
  if (pqdsa == NULL) {
    if (ctx->pkey == NULL) {
      OPENSSL_PUT_ERROR(EVP, EVP_R_NO_PARAMETERS_SET);
      return 0;
    }
    pqdsa = PQDSA_KEY_get0_dsa(ctx->pkey->pkey.pqdsa_key);
  }

  // Caller is getting parameter values.
  if (sig == NULL) {
    if (sig_len != NULL) {
      *sig_len = pqdsa->signature_len;
      return 1;
    }
  }

  if (*sig_len != pqdsa->signature_len) {
    OPENSSL_PUT_ERROR(EVP, EVP_R_BUFFER_TOO_SMALL);
    return 0;
  }

  // Check that the context is properly configured.
  if (ctx->pkey == NULL ||
      ctx->pkey->pkey.pqdsa_key == NULL ||
      ctx->pkey->type != EVP_PKEY_PQDSA) {
    OPENSSL_PUT_ERROR(EVP, EVP_R_OPERATON_NOT_INITIALIZED);
    return 0;
  }

  PQDSA_KEY *key = ctx->pkey->pkey.pqdsa_key;
  if (!key->private_key) {
    OPENSSL_PUT_ERROR(EVP, EVP_R_NO_KEY_SET);
    return 0;
  }

  if (!pqdsa->method->pqdsa_sign(key->private_key, sig, sig_len, message, message_len, NULL, 0)) {
    OPENSSL_PUT_ERROR(EVP, ERR_R_INTERNAL_ERROR);
    return 0;
  }
  return 1;
}

static int pkey_pqdsa_verify_signature(EVP_PKEY_CTX *ctx, const uint8_t *sig,
                                       size_t sig_len, const uint8_t *message,
                                       size_t message_len) {
  PQDSA_PKEY_CTX *dctx = ctx->data;
  const PQDSA *pqdsa = dctx->pqdsa;

  if (pqdsa == NULL) {
    if (ctx->pkey == NULL) {
      OPENSSL_PUT_ERROR(EVP, EVP_R_NO_PARAMETERS_SET);
      return 0;
    }

    pqdsa = PQDSA_KEY_get0_dsa(ctx->pkey->pkey.pqdsa_key);
  }
  // Check that the context is properly configured.
  if (ctx->pkey == NULL ||
    ctx->pkey->pkey.pqdsa_key == NULL ||
    ctx->pkey->type != EVP_PKEY_PQDSA) {
    OPENSSL_PUT_ERROR(EVP, EVP_R_OPERATON_NOT_INITIALIZED);
    return 0;
  }

  PQDSA_KEY *key = ctx->pkey->pkey.pqdsa_key;

  if (sig_len != pqdsa->signature_len ||
      !pqdsa->method->pqdsa_verify(key->public_key, sig, sig_len, message, message_len, NULL, 0)) {
    OPENSSL_PUT_ERROR(EVP, EVP_R_INVALID_SIGNATURE);
    return 0;
  }

  return 1;
}

// Additional PQDSA specific EVP functions.

// This function sets pqdsa parameters defined by |nid| in |pkey|.
int EVP_PKEY_pqdsa_set_params(EVP_PKEY *pkey, int nid) {
  const PQDSA *pqdsa = PQDSA_find_dsa_by_nid(nid);

  if (pqdsa == NULL) {
    OPENSSL_PUT_ERROR(EVP, EVP_R_UNSUPPORTED_ALGORITHM);
    return 0;
  }

  evp_pkey_set_method(pkey, &pqdsa_asn1_meth);

  PQDSA_KEY *key = PQDSA_KEY_new();
  if (key == NULL) {
    // PQDSA_KEY_new sets the appropriate error.
    return 0;
  }

  key->pqdsa = pqdsa;
  pkey->pkey.pqdsa_key = key;

  return 1;
}

// Takes an EVP_PKEY_CTX object |ctx| and sets pqdsa parameters defined
// by |nid|
int EVP_PKEY_CTX_pqdsa_set_params(EVP_PKEY_CTX *ctx, int nid) {
  if (ctx == NULL || ctx->data == NULL) {
    OPENSSL_PUT_ERROR(EVP, ERR_R_PASSED_NULL_PARAMETER);
    return 0;
  }

  // It's not allowed to change context parameters if
  // a PKEY is already associated with the context.
  if (ctx->pkey != NULL) {
    OPENSSL_PUT_ERROR(EVP, EVP_R_INVALID_OPERATION);
    return 0;
  }

  const PQDSA *pqdsa = PQDSA_find_dsa_by_nid(nid);
  if (pqdsa == NULL) {
    OPENSSL_PUT_ERROR(EVP, EVP_R_UNSUPPORTED_ALGORITHM);
    return 0;
  }

  PQDSA_PKEY_CTX *dctx = ctx->data;
  dctx->pqdsa = pqdsa;

  return 1;
}

// Returns a fresh EVP_PKEY object of type EVP_PKEY_PQDSA,
// and sets PQDSA parameters defined by |nid|.
static EVP_PKEY *EVP_PKEY_pqdsa_new(int nid) {
  EVP_PKEY *ret = EVP_PKEY_new();
  if (ret == NULL || !EVP_PKEY_pqdsa_set_params(ret, nid)) {
    EVP_PKEY_free(ret);
    return NULL;
  }

  return ret;
}

EVP_PKEY *EVP_PKEY_pqdsa_new_raw_public_key(int nid, const uint8_t *in, size_t len) {
  if (in == NULL) {
    OPENSSL_PUT_ERROR(EVP, ERR_R_PASSED_NULL_PARAMETER);
    return NULL;
  }

  EVP_PKEY *ret = EVP_PKEY_pqdsa_new(nid);
  if (ret == NULL || ret->pkey.pqdsa_key == NULL) {
    // EVP_PKEY_pqdsa_new sets the appropriate error.
    goto err;
  }

  const PQDSA *pqdsa =  PQDSA_KEY_get0_dsa(ret->pkey.pqdsa_key);
  if (pqdsa->public_key_len != len) {
    OPENSSL_PUT_ERROR(EVP, EVP_R_INVALID_BUFFER_SIZE);
    goto err;
  }

  if (!PQDSA_KEY_set_raw_public_key(ret->pkey.pqdsa_key, in)) {
    // PQDSA_KEY_set_raw_public_key sets the appropriate error.
    goto err;
  }

  return ret;

  err:
    EVP_PKEY_free(ret);
  return NULL;
}

EVP_PKEY *EVP_PKEY_pqdsa_new_raw_private_key(int nid, const uint8_t *in, size_t len) {
  if (in == NULL) {
    OPENSSL_PUT_ERROR(EVP, ERR_R_PASSED_NULL_PARAMETER);
    return NULL;
  }

  EVP_PKEY *ret = EVP_PKEY_pqdsa_new(nid);
  if (ret == NULL || ret->pkey.pqdsa_key == NULL) {
    // EVP_PKEY_kem_new sets the appropriate error.
    goto err;
  }

  const PQDSA *pqdsa =  PQDSA_KEY_get0_dsa(ret->pkey.pqdsa_key);
  if (pqdsa->private_key_len != len) {
    OPENSSL_PUT_ERROR(EVP, EVP_R_INVALID_BUFFER_SIZE);
    goto err;
  }

  if (!PQDSA_KEY_set_raw_private_key(ret->pkey.pqdsa_key, in)) {
    // PQDSA_KEY_set_raw_private_key sets the appropriate error.
    goto err;
  }

  return ret;

  err:
    EVP_PKEY_free(ret);
  return NULL;
}

DEFINE_METHOD_FUNCTION(EVP_PKEY_METHOD, EVP_PKEY_pqdsa_pkey_meth) {
  out->pkey_id = EVP_PKEY_PQDSA;
  out->init = pkey_pqdsa_init;
  out->copy = NULL;
  out->cleanup = pkey_pqdsa_cleanup;
  out->keygen = pkey_pqdsa_keygen;
  out->sign_init = NULL;
  out->sign = NULL;
  out->sign_message = pkey_pqdsa_sign_message;
  out->verify_init = NULL;
  out->verify = NULL;
  out->verify_message = pkey_pqdsa_verify_signature;
  out->verify_recover = NULL;
  out->encrypt = NULL;
  out->decrypt = NULL;
  out->derive = NULL;
  out->paramgen = NULL;
  out->ctrl = NULL;
  out->ctrl_str = NULL;
  out->keygen_deterministic = NULL;
  out->encapsulate_deterministic = NULL;
  out->encapsulate = NULL;
  out->decapsulate = NULL;
}
