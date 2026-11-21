data {
  real y;
}
parameters {
  cholesky_factor_corr[2] p_L;
}
model {
  p_L ~ lkj_corr_cholesky(4.0);
  y ~ normal(p_L[2, 1], 1.0);
}
