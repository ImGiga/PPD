%% Daten Spektrometer 
clc
close all;
clear all;

%% daten einlesen

anzahlverschiedeneMessungen=1;
c=summer(2*anzahlverschiedeneMessungen);

data_sample01 = dlmread('\\bla.txt'); % Geben Sie hier den Pfad der Messdaten an. 

for ww = 1:anzahlverschiedeneMessungen
    k = num2str([ww:ww].','%02d');
    auto.sampleinhalt=eval(sprintf('data_sample%s(:,2)',k));
    if abs(min(auto.sampleinhalt))>abs(max(auto.sampleinhalt))
        auto.sampleinhalt=-auto.sampleinhalt;
    end
    auto.sampleinhalt=auto.sampleinhalt-auto.sampleinhalt(1);
    eval(['auto.sample' num2str(k) '=auto.sampleinhalt']);
    auto.tinhalt=eval(sprintf('data_sample%s(:,1)',k));
    eval(['auto.t' num2str(k) '=auto.tinhalt']);
end

%% calibration
fak=2.9e10;
auto.t01=1e12*auto.t01/fak;

%% Auswertung
figure(1)
hold on;
box on;
set(gca,'FontSize',14) 
xlabel('Verzögerung (ps)');
ylabel('Autokorrelation');
% xlim([0 15]);
% ylim([-6 4]);
for ww = 1:anzahlverschiedeneMessungen
    k = num2str([ww:ww].','%02d');
    plot(eval(sprintf('auto.t%s',k)),eval(sprintf('auto.sample%s',k)), 'LineWidth',1)
end
% legend('Autokorrelation','Location','northeast');
% set(gcf, 'PaperUnits','centimeters','PaperSize',[13 8.8]);
% print('-dpdf', 'ZZZM3fig3.pdf','-fillpage')


% FFT
fft_sample01=fft(auto.sample01);


figure(2)
hold on;
box on;
set(gca,'FontSize',14) 
% xlim([0 200]);
% ylim([0 1e4]);
plot(abs(fft_sample01), 'LineWidth',1);
% legend('Autokorrelation','Location','northeast');
% set(gcf, 'PaperUnits','centimeters','PaperSize',[13 8.8]);
% print('-dpdf', 'ZZZM3fig3.pdf','-fillpage')


cut = ; % Geben Sie hier mithilfe figure(2) den ''cut'' für die hohen Frequenzen an. 
fft_sample01(cut:length(fft_sample01))=zeros(1,length(fft_sample01)-cut+1);
sample01NachFFT=abs(ifft(fft_sample01));

figure(3)
hold on;
xlabel('Verzögerung (ps)');
ylabel('Autokorrelation');
% xlim([0 200]);
% ylim([0 1e4]);
plot(auto.t01,auto.sample01, 'LineWidth',1)
plot(auto.t01,sample01NachFFT, 'LineWidth',3)
% legend('modulierte Autokorrelation','gefilterte Autokorrelation','Location','northeast');
% set(gcf, 'PaperUnits','centimeters','PaperSize',[13 8.8]);
% print('-dpdf', 'ZZZM3fig3.pdf','-fillpage')

% Als nächstes soll an sample01NachFFT eine Gaußkurve gefittet werden. Dies
% können Sie hier in MATLAB tun, oder Sie fügen die Daten in ein anderes
% Programm ein.
T = table(auto.t01,auto.sample01, sample01NachFFT, 'VariableNames',{'Zeit','AutokorrelationVorFFT','AutokorrelationNachFFT'});
writetable(T, 'AutoFFT.txt')



