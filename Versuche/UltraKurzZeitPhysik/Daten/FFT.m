%% Daten Spektrometer 
clc
close all;
clear all;

%% 

anzahlverschiedeneMessungen=2; % Dieses Programm ist dafür vorgesehen, eine Referenzmessung und eine Messung durch eine Probe auszuwerten. Es empfiehlt sich für jede neue Probe ein neues Skrip anzulegen.
c=summer(2*anzahlverschiedeneMessungen);

% daten einlesen

data_sample01 = dlmread('C:\Users\antonia\Documents\Messungen\20211029\Luft  01.dat'); % Geben Sie hier den Pfad der Messdaten an.
data_sample02 = dlmread('C:\Users\antonia\Documents\Messungen\20211029\pWafer  01.dat'); % Geben Sie hier den Pfad der Messdaten an.

for ww = 1:anzahlverschiedeneMessungen
    k = num2str([ww:ww].','%02d');
    pulse.sampleinhalt=eval(sprintf('data_sample%s(:,2)',k));
    eval(['pulse.sample' num2str(k) '=pulse.sampleinhalt']);
    pulse.tinhalt=eval(sprintf('data_sample%s(:,1)',k));
    eval(['pulse.t' num2str(k) '=pulse.tinhalt']);
end


figure(1)
hold on;
box on;
set(gca,'FontSize',14) 
xlabel('Messpunkt');
ylabel('Amplitude');
% xlim([0 2500]);
% ylim([-6 4]);
for ww = 1:anzahlverschiedeneMessungen
    k = num2str([ww:ww].','%02d');
    plot(eval(sprintf('pulse.sample%s',k)), 'LineWidth',3)
end
% legend('Luft','pWafer','Location','northeast');
% set(gcf, 'PaperUnits','centimeters','PaperSize',[13 8.8]);
% print('-dpdf', 'ZZZM3fig3.pdf','-fillpage')

 
% Finden Sie mithilfe figure(1) heraus ab welchem Messpunkt der
% Etalon-Effekt auftritt und geben Sie diesen Wert als cut01 bzw. cut02 an,
% damit die Reflektionen abgeschnitten werden. Wenn es keine Reflektionen
% gibt, geben Sie den letzten Messpunkt an.
cut01=;
cut02=;

for ww = 1:anzahlverschiedeneMessungen
    k = num2str([ww:ww].','%02d');
    pulse.samplenamek=eval(sprintf('pulse.sample%s',k));
    pulse.tnamek=eval(sprintf('pulse.t%s',k));
    cut=eval(sprintf('cut%s',k));
    pulse.samplenamek=pulse.samplenamek-pulse.samplenamek(1);
    pulse.samplenamek(cut:end)=[];
    pulse.tnamek(cut:end)=[];
    eval([sprintf('pulse.sample%s',k) '=pulse.samplenamek']);
    eval([sprintf('pulse.t%s',k) '=pulse.tnamek']);
end


figure(2)
hold on;
box on;
set(gca,'FontSize',14) 
xlabel('Verzögerung (ps)');
ylabel('Amplitude');
% xlim([0 15]);
% ylim([-6 4]);
for ww = 1:anzahlverschiedeneMessungen
    k = num2str([ww:ww].','%02d');
    plot(eval(sprintf('pulse.t%s',k)),eval(sprintf('pulse.sample%s',k)), 'LineWidth',3)
end
% legend('Luft','pWafer','Location','northeast');
% set(gcf, 'PaperUnits','centimeters','PaperSize',[13 8.8]);
% print('-dpdf', 'ZZZM3fig3.pdf','-fillpage')


% Fensterfunktion, um Artefakte der Auswertung zu verhindern.
for w = 1:anzahlverschiedeneMessungen
    k = num2str([w:w].','%02d');
   sampleinhalt=eval(sprintf('pulse.sample%s',k));
   tinhalt=eval(sprintf('pulse.t%s',k));
    sampleinhaltzw=0;
    b=0;
    tFenster=0;
    for kk=1:length(sampleinhalt)
        b=b+(tinhalt(3)-tinhalt(2));
        tFenster(kk)=b;
    end
    
    a=find(sampleinhalt==min(sampleinhalt));
    diff=round(2*(abs(length(sampleinhalt)*0.5-a)));
    sampleinhaltzw(1:diff(1))=zeros(1,diff(1));
    sampleinhaltzw(diff(1):length(sampleinhalt)+diff(1)-1)=sampleinhalt;
    sampleinhalt=sampleinhaltzw;
    cuthanbla=length(sampleinhalt)-3000;
    sampleinhalt(length(sampleinhalt)-cuthanbla:end)=[];
    sampleinhalt(1:cuthanbla)=[];
    
    fensterhan=hann(length(sampleinhalt));
    sampleinhalthan=sampleinhalt.*fensterhan';

    langehan=length(sampleinhalthan);
    langehan2=5000;
    sampleinhalthan(langehan:langehan2)=zeros(1,langehan2-langehan+1);
    fensterhan(langehan:langehan2)=zeros(1,langehan2-langehan+1);
    
    b=0;
    tFensterzw=tFenster;
    tFenster=0;
    for kk=1:length(sampleinhalthan)
        b=b+(tFensterzw(3)-tFensterzw(2));
        tFenster(kk)=b;
    end
      
%     figure(13)
%     plot(tFenster,sampleinhalthan, 'LineWidth',1);
%     plot(tFenster,fensterhan, 'LineWidth',1);

    eval([sprintf('fenster.sample%s',k) '=sampleinhalthan']);
    eval([sprintf('fenster.t%s',k) '=tFenster']);

end

% FFT
for ww = 1:anzahlverschiedeneMessungen
    k = num2str([ww:ww].','%02d');
    ffd.sampleinhalt=eval(sprintf('fenster.sample%s',k));
    ffd.tinhalt=eval(sprintf('fenster.t%s',k));

    % Definitions
    Fs=1/(ffd.tinhalt(2)-ffd.tinhalt(1)); %sampling freq
    N=length(ffd.sampleinhalt);
    Nfft=2^nextpow2(N);
    f=Fs/2*linspace(0,1,1+Nfft/2); % create freqs vector

    % main
    fft_samplezw=fft(ffd.sampleinhalt,Nfft)/N; % perform fft transform
    fft_sample=fft_samplezw(1:1+Nfft/2);
    freq=f;
    
    eval([sprintf('ffd.sample%s',k) '=fft_sample']);
    eval([sprintf('ffd.freq%s',k) '=freq']);
end

betrag.FFTsample01=abs(ffd.sample01);
betrag.FFTsample02=abs(ffd.sample02);

theo=; % Geben Sie hier für Aufgabe 4.2.2 den in Frage 7 berechneten Wert an.


figure(3)
hold on;
box on;
set(gca,'FontSize',14) 
xlabel('Frequenz (THz)');
ylabel('Transmission');
% plot(ffd.freq01,Probe./Referenz); 
line([ffd.freq01(1) ffd.freq01(end)], [theo theo],'color', [0 0 0],'Linewidth',3) 
xlim([0.4 2.25]);
ylim([0 1.1]);
% legend('pWafer Spektrum','Location','northeast');
% set(gcf, 'PaperUnits','centimeters','PaperSize',[13 8.8]);
% print('-dpdf', 'ZZZM3fig3.pdf','-fillpage')



