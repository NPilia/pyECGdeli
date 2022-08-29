clear
load('results_ecg_deli.mat')

errors2 = all_annotations(:,2);
errors1 = all_annotations(:,1);

measurements = cellfun(@(x) x{1,1}(:,6), all_FPT_cell, 'UniformOutput', false);

err = nan(10000,1);
k=1;
for sig = 1:1:length(all_FPT_cell)
    for ann = 1:1:size(all_annotations{sig,1},1)
        errCh1=all_FPT_cell{sig,1}{1,1}(:,6)-double(all_annotations{sig,1}(ann,2));
        errCh2=all_FPT_cell{sig,1}{1,2}(:,6)-double(all_annotations{sig,1}(ann,2));
        [~,posCh1]=min(abs(errCh1));
        [~,posCh2]=min(abs(errCh2));
        [~,b]=min([abs(errCh1(posCh1)),abs(errCh2(posCh2))]);
        if b==1
            err(k,1)=errCh1(posCh1);
        elseif b==2
            err(k,1)=errCh2(posCh2);
        end
        k = k + 1;
    end
    if size(all_annotations{sig,2},1)>1
        for ann = 1:1:size(all_annotations{sig,2},1)
            errCh1=all_FPT_cell{sig,1}{1,1}(:,6)-double(all_annotations{sig,2}(ann,2));
            errCh2=all_FPT_cell{sig,1}{1,2}(:,6)-double(all_annotations{sig,2}(ann,2));
            [~,posCh1]=min(abs(errCh1));
            [~,posCh2]=min(abs(errCh2));
            [~,b]=min([abs(errCh1(posCh1)),abs(errCh2(posCh2))]);
            if b==1
                err(k,1)=errCh1(posCh1);
            elseif b==2
                err(k,1)=errCh2(posCh2);
            end
            k = k + 1;
        end
    end
end

err = err(~isnan(err));

mean(err(:,1))
std(err(:,1))

mean(abs(err(:,1)))
std(abs(err(:,1)))

median(abs(err(:,1)))
iqr(abs(err(:,1)))

%% QRSon
err = nan(10000,1);
k=1;
for sig = 1:1:length(all_FPT_cell)
    for ann = 1:1:size(all_annotations{sig,1},1)
        if double(all_annotations{sig,1}(ann,1))~=-1
            errCh1=all_FPT_cell{sig,1}{1,1}(:,6)-double(all_annotations{sig,1}(ann,1));
            errCh2=all_FPT_cell{sig,1}{1,2}(:,6)-double(all_annotations{sig,1}(ann,1));
            [~,posCh1]=min(abs(errCh1));
            [~,posCh2]=min(abs(errCh2));
            [~,b]=min([abs(errCh1(posCh1)),abs(errCh2(posCh2))]);
            if b==1
                err(k,1)=errCh1(posCh1);
            elseif b==2
                err(k,1)=errCh2(posCh2);
            end
            k = k + 1;
        end
    end
    if size(all_annotations{sig,2},1)>1
        for ann = 1:1:size(all_annotations{sig,2},1)
            if double(all_annotations{sig,1}(ann,1))~=-1
                errCh1=all_FPT_cell{sig,1}{1,1}(:,6)-double(all_annotations{sig,2}(ann,1));
                errCh2=all_FPT_cell{sig,1}{1,2}(:,6)-double(all_annotations{sig,2}(ann,1));
                [~,posCh1]=min(abs(errCh1));
                [~,posCh2]=min(abs(errCh2));
                [~,b]=min([abs(errCh1(posCh1)),abs(errCh2(posCh2))]);
                if b==1
                    err(k,1)=errCh1(posCh1);
                elseif b==2
                    err(k,1)=errCh2(posCh2);
                end
                k = k + 1;
            end
        end
    end
end

err = err(~isnan(err));

mean(err(:,1))
std(err(:,1))

mean(abs(err(:,1)))
std(abs(err(:,1)))

median(abs(err(:,1)))
iqr(abs(err(:,1)))



%% QRSoff
err = nan(10000,1);
k=1;
for sig = 1:1:length(all_FPT_cell)
    for ann = 1:1:size(all_annotations{sig,1},1)
        if double(all_annotations{sig,1}(ann,3))~=-1
            errCh1=all_FPT_cell{sig,1}{1,1}(:,6)-double(all_annotations{sig,1}(ann,3));
            errCh2=all_FPT_cell{sig,1}{1,2}(:,6)-double(all_annotations{sig,1}(ann,3));
            [~,posCh1]=min(abs(errCh1));
            [~,posCh2]=min(abs(errCh2));
            [~,b]=min([abs(errCh1(posCh1)),abs(errCh2(posCh2))]);
            if b==1
                err(k,1)=errCh1(posCh1);
            elseif b==2
                err(k,1)=errCh2(posCh2);
            end
            k = k + 1;
        end
    end
    if size(all_annotations{sig,2},1)>1
        for ann = 1:1:size(all_annotations{sig,2},1)
            if double(all_annotations{sig,1}(ann,3))~=-1
                errCh1=all_FPT_cell{sig,1}{1,1}(:,6)-double(all_annotations{sig,2}(ann,3));
                errCh2=all_FPT_cell{sig,1}{1,2}(:,6)-double(all_annotations{sig,2}(ann,3));
                [~,posCh1]=min(abs(errCh1));
                [~,posCh2]=min(abs(errCh2));
                [~,b]=min([abs(errCh1(posCh1)),abs(errCh2(posCh2))]);
                if b==1
                    err(k,1)=errCh1(posCh1);
                elseif b==2
                    err(k,1)=errCh2(posCh2);
                end
                k = k + 1;
            end
        end
    end
end

err = err(~isnan(err));

mean(err(:,1))
std(err(:,1))

mean(abs(err(:,1)))
std(abs(err(:,1)))

median(abs(err(:,1)))
iqr(abs(err(:,1)))

