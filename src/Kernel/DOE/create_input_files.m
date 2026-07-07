function create_input_files(templateFile, outputFolder, DOE)

if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

CT_txt = fileread(templateFile(1));
SD_txt = fileread(templateFile(2));

N = size(DOE,1);

for i = 1:N

    h1   = DOE(i,1);
    h2   = DOE(i,2);
    mu   = DOE(i,3);

    % convert to 2-digit integers
    h1s = floor(h1);
    h2s = floor(h2);
    mus = floor(mu*100);

    CT_newtxt = CT_txt;
    SD_newtxt = SD_txt;

    CT_newtxt = strrep(CT_newtxt, '<HTC1>', num2str(h1));
    CT_newtxt = strrep(CT_newtxt, '<HTC2>', num2str(h2));

    filename1 = sprintf('%d-CT-CS-%02d-%02d-%02d.inp', ...
                    i, h1s, h2s, mus);

    SD_newtxt = strrep(SD_newtxt, '<FRIC>', num2str(mu));


    filename2 = sprintf('%d-SD-CS-%02d-%02d-%02d.inp', ...
                    i, h1s, h2s, mus);


    odbName = strrep(filename1,'.inp','.odb');
    SD_newtxt = strrep(SD_newtxt,'<CT-job-name>',odbName);

    % write file

    fprintf('%s\n', filename1);

    outFile = fullfile(outputFolder, filename1);
    fid = fopen(outFile,'w');
    fwrite(fid, CT_newtxt);
    fclose(fid);

    outFile = fullfile(outputFolder, filename2);
    fid = fopen(outFile,'w');
    fwrite(fid, SD_newtxt);
    fclose(fid);

end

end