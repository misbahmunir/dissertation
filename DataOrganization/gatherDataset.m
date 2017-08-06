% this file gets the information about all the video files in a folder...
% get the dataset path first...

path = input('enter the dataset main folder path: ', 's');
contentlist = getDirListing( path );

% clean the list to remove all rar files...
j = 1;
newlength = length(contentlist);
while(j<=newlength)
    if (contentlist(j).isdir == 0)
        contentlist(j) = '';
    else
        j = j+1;
    end
    newlength = length(contentlist);
end

% get the subdirectories of the list...
l = length(contentlist);
newList = [];
i = 1;
traindata = [];
testdata = [];
valdata = [];
tr = 1; te = 1; ve = 1;

while (i <= l)
    disp(i);
    if (isdir(contentlist(i).name))
        temp = getDirListing(contentlist(i).name);
        xx = length(temp);
        for x = 1:length(temp)
            if (x <= 0.7*xx)
                traindata(tr).path = temp(x).name;
                traindata(tr).label = i;
                tr = tr+1;
            else
                if(rem(x ,2) == 1)
                    testdata(te).path = temp(x).name;
                	testdata(te).label = i;
                    te = te+1;
                else
                    valdata(ve).path = temp(x).name;
                	valdata(ve).label = i;
                    ve = ve + 1;
                end
            end
        end
%         t = t+x;
    end
    i = i+1;
end

% write data on the file...
writeData('ucfs_test.txt', testdata);
writeData('ucfs_val.txt', valdata);
writeData('ucfs_train.txt', traindata);