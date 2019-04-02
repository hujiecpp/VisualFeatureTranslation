function x = crow_act(x)

  if ~max(size(x, 1), size(x, 2))
    x = zeros(size(x, 3), 1, class(x));
    return;
  end

  % compute_crow_spatial_weight S
  a = 2;
  b = 2;

  S = sum(x, 3);
  z = (sum(sum(S.^a)))^(1.0/a);
  S = (S ./ z).^(1.0/b);

  % compute_crow_channel_weight C
  sz = size(x);
  w = sz(1);
  h = sz(2);
  K = sz(3);
  area = 1.0 * w * h;
  C = zeros(K, 1);

  for i = 1 : K
    tmp = x(:, :, i);
    C(i, 1) = sum(sum(tmp ~= 0)) / area;
  end

  C_sum = sum(C);
  for i = 1 : K
    d = C(i, 1);
    if d > 0
        C(i, 1) = log(C_sum / d);
    else
        C(i, 1) = 0;
    end
  end

  for i = 1 : K
    x(:, :, i) = x(:, :, i) .* S;
  end
  x = reshape(sum(sum(x, 1), 2), [size(x,3) 1]);
  x = x .* C;