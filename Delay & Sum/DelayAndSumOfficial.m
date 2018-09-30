clc; clear;

%% Input data
% Every single track of the all microphones is read and stored.

for i = 1:16
    if i<10
        [s(i,:),fs] = audioread(['porta+applausoAttenuato/' '0' num2str(i) 'sn.wav']);
    else
        [s(i,:),~] = audioread(['porta+applausoAttenuato/' num2str(i) 'sn.wav']);
    end
end

%% STFT 
% In this section for each microphone compute the STFT 

wlen = 1024;
hop = 256;
nftt = 1024;

for i = 1:16
    
    [stfts(:,:,i),freq] = stft(s(i,:),wlen,hop,nftt,fs);
end


%% Filter array
% in this section the near field filter is created in function of the
% distance between each microphone and the source signal

M = 16;     % number of mic
x = 0.06;   % mic distance
base_dist = x/2;
c = 340;    % sound speed

% caso Far Field Delay & Sum

%freq = linspace(0, fs, wlen);
%freq=freq;
%teta = (teta/360)*2*pi;
% for i = 0:7
%     for f = 1:length(freq)
%         h(f,9+i) = exp(-1j*2*pi*freq(1,f)*cos(teta)*(base_dist+i*x)/c)/M;
%         h(f,8-i) = exp(-1j*2*pi*freq(1,f)*cos(teta)*(-base_dist-i*x)/c)/M;
%     end
% end
% teta = atan2(src_pos(2),src_pos(1));
% for f = 1:length(freq)
%     steering=exp(-1j*2*pi*freq(f)/c*(cos(teta)*mic_pos(1,:)'));
%     h(f,:) = (steering./(steering'*steering)).';
% end

mic_pos=0:x:x*(M-1);
mic_pos=[mic_pos-x*(M-1)/2;zeros(1,M)];

src_pos=[0.25,0.7]';

dist=pdist2(mic_pos',src_pos');
for f = 1:length(freq)
    steering=exp(-1j*2*pi*freq(f)/c*dist)./(4*pi*dist);
    h(f,:) = (steering./(steering'*steering)).';
end

%% Filtering each frequency for each time frame

hN = reshape(h,size(h,1),1,size(h,2));
h = reshape(h,size(h,1),1,size(h,2));
h = repmat(h,1,size(stfts,2),1);
filt_sound_rep = conj(h).*stfts;
sums = sum(filt_sound_rep,3);

%% Istft
% calculate the ISTFT to obtain a signal in time domain

[signal_ds, t] = istft(sums,hop,nftt,fs);

%% test 
final = audioplayer(signal_ds,fs);

%% Noramlize and save the result

signal_ds=0.99*(signal_ds./max(abs(signal_ds)));
audiowrite('sn.wav',signal_ds,fs);
%audiowrite('1min_noise.wav',noise_ds,fs);




