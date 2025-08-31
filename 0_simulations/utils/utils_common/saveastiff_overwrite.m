% saveastiff, forced overwrite

%%% ELiiiiiii, 20250315
function res = saveastiff_overwrite(data, path, options)

    options.overwrite = true;
    res = saveastiff(data,path,options);