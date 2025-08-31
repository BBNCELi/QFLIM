% Generate boolean values based on specified possibility
% e.g. when possib = 0.1, run randpossib 100 times (with siz=1) it exports ~10 TRUEs
%%ELiiiiiii, 20240117
%%ELiiiiiii, 20240119, add when possib is not a scalar but a matrix
function bool = randpossib(possib, siz)
%% output size
if isscalar(possib)
    if nargin < 2
        siz = 1;
    end
    if possib < 0 || possib > 1
        warning('Possibility in randpossib seems obsurd');
    end
else
    if nargin == 2
        error('Do not accept input size when possib is not a scalar');
    end
    siz = size(possib);
end

%% seed
rng('shuffle');

%% rand
randomValues = rand(siz);
bool = randomValues < possib;