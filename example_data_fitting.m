%Underfitting & Overfitting examples

t = [-1 -0.5 0 0.5 1];

A1 = a_matrix(t, 5, 1);
A2 = a_matrix(t, 5, 2);
A3 = a_matrix(t, 5, 3);
A4 = a_matrix(t, 5, 4);
A5 = a_matrix(t, 5, 5);

b = [1;
    0.5;
    0;
    0.5;
    2];

x1 = x_matrix(A1, b);
x2 = x_matrix(A2, b);
x3 = x_matrix(A3, b);
x4 = x_matrix(A4, b);
x5 = x_matrix(A5, b);

t1 = linspace(-1,1, 200);
plot(t1, polyval(transpose(x1), t1), ...
    t1, polyval(transpose(x2), t1), ...
    t1, polyval(transpose(x3), t1), ...
    t1, polyval(transpose(x4), t1), ...
    t1, polyval(transpose(x5), t1));
hold on;
plot(t, b, '.', 'markersize', 10);
hold off;

xlim([-1.5 1.5])

function a = a_matrix(t, n_row, n_col)
    a = zeros(n_row, n_col);
    for j = 1:n_col
        for i = 1:n_row
            a(i, j) = (t(i))^(j-1);
        end
    end
end

function x = x_matrix(a_matrix, b)
   x = flip(inv(transpose(a_matrix)*a_matrix)*transpose(a_matrix)*b);
end
