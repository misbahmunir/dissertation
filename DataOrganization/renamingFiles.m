% Code Reference: https://uk.mathworks.com/matlabcentral/answers/1760-how-to-rename-a-bunch-of-files-in-a-folder

mainPath = '/mnt/sun-gamma/mm-workspace/Dissertation/data/hmdb51_org';
subDirectories = dir(mainPath);
l = length(subDirectories);
i = 1;
while i < l
    if (strcmp(subDirectories(i).name, '.'))
        subDirectories(i) = '';
        l = l-1;
    elseif (strcmp(subDirectories(i).name, '..'))
        subDirectories(i) = '';
        l = l-1;
    else
        i = i+1;
    end
end

for i = 1:length(subDirectories)
%    oldName =  [mainPath '/' subDirectories(i).name '/*.avi']
    oldnames = dir([mainPath '/' subDirectories(i).name '/*.avi'])
    for j = 1:length(oldnames)
        movefile([mainPath '/' subDirectories(i).name '/' oldnames(j).name], [mainPath '/' subDirectories(i).name '/' sprintf('%d%03d.avi', i, j)])
    end
end