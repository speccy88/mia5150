"""Normalized multi-positive contrastive objective and known-positive filtering."""
def negative_ids(ranked,positive_ids,eligible):
    positives=set(positive_ids);allowed=set(eligible)
    return [sid for sid in ranked if sid in allowed and sid not in positives]

def contrastive_loss(query,documents,positive_mask,temperature=.05,candidate_mask=None):
    import torch
    from torch.nn.functional import normalize
    assert temperature>0
    logits=normalize(query.float(),dim=-1)@normalize(documents.float(),dim=-1).T/temperature
    positive_mask=positive_mask.to(device=logits.device,dtype=torch.bool)
    valid=torch.ones_like(positive_mask) if candidate_mask is None else candidate_mask.to(device=logits.device,dtype=torch.bool)
    assert logits.shape==positive_mask.shape==valid.shape
    assert (positive_mask & valid).any(dim=1).all()
    assert not (positive_mask & ~valid).any()
    denominator=logits.masked_fill(~valid,-torch.inf).logsumexp(dim=1)
    numerator=logits.masked_fill(~positive_mask,-torch.inf).logsumexp(dim=1)
    return (denominator-numerator).mean()
