set(gcf, 'InvertHardCopy', 'off', 'color','w');

saveFileName = [fileName '_' con1 '_' con2 '_' listType '.png'];

%saveas(gcf, [outPath  '\' saveFileName])
print(fullfile(outPath, saveFileName),'-dpng','-r163')
