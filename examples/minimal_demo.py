"""CPU-only demonstration of the three inference-time method components."""

import numpy as np

from llm_tse_grounding import (
    CDCS5_CANDIDATES,
    VOCAB_SIZE,
    refine_gnr,
    select_by_enrollment_similarity,
    select_csg_tokens,
)


scores = {
    name: score
    for name, score in zip(CDCS5_CANDIDATES, [0.71, 0.68, 0.70, 0.69, 0.79])
}
selection = select_by_enrollment_similarity(scores)
print("CDCS-5 selection:", selection)

evidence = np.asarray([0, 1, 2], dtype=np.int64)
logits = np.zeros((3, VOCAB_SIZE), dtype=np.float32)
logits[:, VOCAB_SIZE - 1] = 2.0
print("CSG tokens:", select_csg_tokens(logits, evidence, grounding_lambda=1.0))

anchor = np.asarray([0, 0, 0], dtype=np.int64)
teacher_forced_logits = np.zeros((3, VOCAB_SIZE), dtype=np.float32)
teacher_forced_logits[:, 1] = 3.0
result = refine_gnr(teacher_forced_logits, anchor, top_k=20, radius=2)
print("GNR tokens:", result.tokens)
print("GNR edit rate:", result.edit_rate)
