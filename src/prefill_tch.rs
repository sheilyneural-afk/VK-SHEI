//! configured-model query projection into M-SLIM's parallel search dimension.
//!
//! Every Q head is preserved independently and mapped only to the KV head
//! that serves it. This avoids cancellation of a rare but important query.

use anyhow::{Result, ensure};
use tch::{Kind, Tensor};

pub const Q_HEADS: usize = 16;
pub const KV_HEADS: usize = 2;
pub const GROUP_SIZE: usize = Q_HEADS / KV_HEADS;
pub const HEAD_DIM: usize = 128;
pub const PLANNER_DIM: usize = KV_HEADS * HEAD_DIM;

#[derive(Clone, Debug, PartialEq)]
pub struct ProtectedQuery {
    pub query_head: usize,
    pub query_token_offset: usize,
    pub planner_vector: Vec<f32>,
}

pub fn protected_queries(query: &Tensor) -> Result<Vec<ProtectedQuery>> {
    let shape = query.size();
    ensure!(
        shape.len() == 4
            && shape[0] == 1
            && shape[1] == Q_HEADS as i64
            && shape[2] > 0
            && shape[3] == HEAD_DIM as i64,
        "protección esperaba Q [1,16,tokens,128], recibió {shape:?}"
    );
    let query = query.to_kind(Kind::Float);
    let query_tokens = shape[2] as usize;
    let mut protected = Vec::with_capacity(Q_HEADS * query_tokens);
    for token in 0..query_tokens {
        for head in 0..Q_HEADS {
            let slice = query
                .narrow(1, head as i64, 1)
                .narrow(2, token as i64, 1)
                .contiguous();
            let mut values = vec![0_f32; HEAD_DIM];
            slice.copy_data(&mut values, HEAD_DIM);
            ensure!(
                values.iter().all(|value| value.is_finite()),
                "consulta no finita"
            );
            let mut planner_vector = vec![0.0_f32; PLANNER_DIM];
            let start = (head / GROUP_SIZE) * HEAD_DIM;
            planner_vector[start..start + HEAD_DIM].copy_from_slice(&values);
            protected.push(ProtectedQuery {
                query_head: head,
                query_token_offset: token,
                planner_vector,
            });
        }
    }
    Ok(protected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_q_head_maps_only_to_its_serving_kv_head() {
        let query = Tensor::zeros([1, 16, 2, 128], (Kind::Float, tch::Device::Cpu));
        for head in 0..Q_HEADS {
            let _ = query
                .get(0)
                .get(head as i64)
                .get(1)
                .fill_((head + 1) as f64);
        }
        let protected = protected_queries(&query).unwrap();
        assert_eq!(protected.len(), Q_HEADS * 2);
        for item in protected.iter().filter(|item| item.query_token_offset == 1) {
            let serving_head = item.query_head / GROUP_SIZE;
            for (index, value) in item.planner_vector.iter().enumerate() {
                assert_eq!(
                    *value,
                    if index / HEAD_DIM == serving_head {
                        (item.query_head + 1) as f32
                    } else {
                        0.0
                    }
                );
            }
        }
    }
}
