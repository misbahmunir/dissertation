function [] = writeData(filename, dataelements)
%writeData writes the path and label data on a text file specified by the
%user...

fileID = fopen(filename,'w');
for i = 1:length(dataelements)
    fprintf(fileID, '%s %u\n', dataelements(i).path, dataelements(i).label);
end
fclose(fileID);

end

