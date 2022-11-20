function k = KappaCoefficient(Confiusion_Matrix)

M = sum(sum(Confiusion_Matrix));
nij = trace(Confiusion_Matrix);
ni = sum(Confiusion_Matrix')';
nj = sum(Confiusion_Matrix);

ninj = nj*ni;

k = (M*nij - ninj)/(M^2 - ninj);

end