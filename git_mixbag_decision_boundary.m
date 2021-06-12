clc
clear 

load('engent_train');
load('engent_test');
%%
X=engent_train(1:140,[2,3]);    %first and second features (total features; 40)
Y=engent_train(1:140,1);
[N,D]=size(X);
%%
res=[];
res_past_acc=[];

bestInd=[];  
past_acc=0;
for k=1:10
temp=[];
Nbag=5;
Nuse=uint8(N*.65);
classifiers=cell(1,Nbag);

for i=1:Nbag
    ind=ceil(N*rand(Nuse,1));
    Xi=X(ind,:);   % :
    Yi=Y(ind);   % :
    if i==1
    classifiers{i}=fitctree(Xi,Yi,'AlgorithmForCategorical','OVAbyClass');
    elseif i==2
    classifiers{i}=fitcknn(Xi,Yi,'NumNeighbors',13,'Distance','cosine');
    elseif i==3
    classifiers{i}=fitcdiscr(Xi,Yi,'DiscrimType','linear');
    elseif i==4
    classifiers{i}=fitcnb(Xi,Yi);
    elseif i==5
    classifiers{i}=fitcsvm(Xi,Yi,'KernelFunction','linear','BoxConstraint',1,'ClassNames',[1,2]); 
    else
      break
    end
    temp=[temp,ind]; 
end
Xtest=engent_test(1:140,2:3);  %2:end
Ytest=engent_test(1:140,1);
[Ntest,D]=size(Xtest);  %D
predict1=zeros(Ntest,Nbag);
for i= 1:Nbag,
     predict1(:,i)=predict(classifiers{i},Xtest);
end;
predict1=(mean(predict1,2)>1.5)+1;
[c,order]=confusionmat(Ytest, predict1);
acc=sum(diag(c)/sum(c(:)));
if past_acc < acc
    bestInd=temp;
    past_acc=acc;
end
res=[res acc];
res_past_acc=[res_past_acc past_acc];
end
fprintf('Max:%f\tAvg.:%f\tStd.:%f\n',max(res),mean(res),std(res));
fprintf('Best:%f\n',max(res_past_acc));
%%
classifier_name = {'Tree','KNN','Discr','NB','SVM'};

x1range = min(X(:,1)):.01:max(X(:,1));    %% possible :1 as x=2:end
x2range = min(X(:,2)):.01:max(X(:,2));
[xx1, xx2] = meshgrid(x1range,x2range);
XGrid = [xx1(:) xx2(:)];

for i = 1:numel(classifiers)
   predictedspecies = predict(classifiers{i},XGrid);

   subplot(3,2,i);
   alpha=0.2;
   gscatter(xx1(:),xx2(:), predictedspecies,'rg');
   
   title(classifier_name{i})
   legend off, axis tight

end
legend('1-Left','2-Right','Location',[0.35,0.01,0.35,0.05],'Orientation','horizontal','FontSize',8,'FontWeight','bold');
%%

subplot(3,2,6);
gscatter(Xtest(:,1),Xtest(:,2),Ytest,'rg','sd');
xlabel('Feature-1');
ylabel('Feature-2');