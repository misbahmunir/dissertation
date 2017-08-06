function [list] = getDirListing( path )
%This method gets the list of all the files and folder names at the
%provided path..

list = dir(path);

% clean the list...
l = length(list);
i = 1;
while i < l
    if (strcmp(list(i).name, '.'))
        list(i) = '';
        l = l-1;
    elseif (strcmp(list(i).name, '..'))
        list(i) = '';
        l = l-1;
    else
        i = i+1;
    end
end

% list(1) = '';
% list(1) = '';

    for j = 1:length(list)
        list(j).name = strcat(path, '/', list(j).name);
    end

end

