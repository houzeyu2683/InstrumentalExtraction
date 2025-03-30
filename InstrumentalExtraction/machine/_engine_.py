import os
import math
import torch
import librosa
import numpy
import threading
import tqdm
import scipy.io
import InstrumentalExtraction.architecture

parameter = {
    'bins': 672, 'unstable_bins': 8, 'reduction_bins': 637, 
    'band': {
        1: {
            'sr': 7350, 'hl': 80, 'n_fft': 640, 'crop_start': 0, 'crop_stop': 85, 
            'lpf_start': 25, 'lpf_stop': 53, 'res_type': 'polyphase'
        }, 
        2: {
            'sr': 7350, 'hl': 80, 'n_fft': 320, 'crop_start': 4, 'crop_stop': 87, 
            'hpf_start': 25, 'hpf_stop': 12, 'lpf_start': 31, 'lpf_stop': 62, 
            'res_type': 'polyphase'
        }, 
        3: {
            'sr': 14700, 'hl': 160, 'n_fft': 512, 'crop_start': 17, 'crop_stop': 216, 
            'hpf_start': 48, 'hpf_stop': 24, 'lpf_start': 139, 'lpf_stop': 210, 
            'res_type': 'polyphase'
        }, 
        4: {
            'sr': 44100, 'hl': 480, 'n_fft': 960, 'crop_start': 78, 'crop_stop': 383, 
            'hpf_start': 130, 'hpf_stop': 86, 'res_type': 'kaiser_fast'
        }
    }, 
    'sr': 44100, 'pre_filter_start': 668, 'pre_filter_stop': 672, 
    'mid_side': False, 'mid_side_b': False, 'mid_side_b2': False, 
    'stereo_w': False, 'stereo_n': False, 'reverse': False
}

'''in process

'''

def wave_to_spectrogram_mt(wave, hop_length, n_fft, mid_side=False, mid_side_b2=False, reverse=False):
    if reverse:
        wave_left = numpy.flip(numpy.asfortranarray(wave[0]))
        wave_right = numpy.flip(numpy.asfortranarray(wave[1]))
    elif mid_side:
        wave_left = numpy.asfortranarray(numpy.add(wave[0], wave[1]) / 2)
        wave_right = numpy.asfortranarray(numpy.subtract(wave[0], wave[1]))
    elif mid_side_b2:
        wave_left = numpy.asfortranarray(numpy.add(wave[1], wave[0] * .5))
        wave_right = numpy.asfortranarray(numpy.subtract(wave[0], wave[1] * .5))
    else:
        wave_left = numpy.asfortranarray(wave[0])
        wave_right = numpy.asfortranarray(wave[1])
   
    def run_thread(**kwargs):
        global spec_left
        spec_left = librosa.stft(**kwargs)

    thread = threading.Thread(target=run_thread, kwargs={'y': wave_left, 'n_fft': n_fft, 'hop_length': hop_length})
    thread.start()
    spec_right = librosa.stft(wave_right, n_fft=n_fft, hop_length=hop_length)
    thread.join()   
    
    spec = numpy.asfortranarray([spec_left, spec_right])

    return spec

def fft_lp_filter(spec, bin_start, bin_stop):
    g = 1.0
    for b in range(bin_start, bin_stop):
        g -= 1 / (bin_stop - bin_start)
        spec[:, b, :] = g * spec[:, b, :]
        
    spec[:, bin_stop:, :] *= 0

    return spec

def combine_spectrograms(specs):
    l = min([specs[i].shape[2] for i in specs])    
    spec_c = numpy.zeros(shape=(2, parameter['bins'] + 1, l), dtype=numpy.complex64)
    offset = 0
    bands_n = len(parameter['band'])
    
    for d in range(1, bands_n + 1):
        h = parameter['band'][d]['crop_stop'] - parameter['band'][d]['crop_start']
        spec_c[:, offset:offset+h, :l] = specs[d][:, parameter['band'][d]['crop_start']:parameter['band'][d]['crop_stop'], :l]
        offset += h
        
    if offset > parameter['bins']:
        raise ValueError('Too much bins')
        
    # lowpass fiter
    if parameter['pre_filter_start'] > 0: # and mp.param['band'][bands_n]['res_type'] in ['scipy', 'polyphase']:   
        if bands_n == 1:
            spec_c = fft_lp_filter(spec_c, parameter['pre_filter_start'], parameter['pre_filter_stop'])
        else:
            gp = 1        
            for b in range(parameter['pre_filter_start'] + 1, parameter['pre_filter_stop']):
                g = math.pow(10, -(b - parameter['pre_filter_start']) * (3.5 - gp) / 20.0)
                gp = g
                spec_c[:, b, :] *= g
                
    return numpy.asfortranarray(spec_c)

def make_padding(width, cropsize, offset):
    left = offset
    roi_size = cropsize - left * 2
    if roi_size == 0:
        roi_size = cropsize
    right = roi_size - (width % roi_size) + left

    return left, right, roi_size

def inference(X_spec, device, model, aggressiveness, data):
    '''
    data ： dic configs
    '''
    
    def _execute(X_mag_pad, roi_size, n_window, device, model, aggressiveness,is_half=True):
        model.eval()
        with torch.no_grad():
            preds = []
            
            iterations = [n_window]

            total_iterations = sum(iterations)            
            for i in tqdm.tqdm(range(n_window), total=total_iterations): 
                start = i * roi_size
                X_mag_window = X_mag_pad[None, :, :, start:start + data['window_size']]
                X_mag_window = torch.from_numpy(X_mag_window)
                if(is_half==True):X_mag_window=X_mag_window.half()
                X_mag_window=X_mag_window.to(device)

                pred = model.predict(X_mag_window, aggressiveness)

                pred = pred.detach().cpu().numpy()
                preds.append(pred[0])
                
            pred = numpy.concatenate(preds, axis=2)
        return pred
    
    def preprocess(X_spec):
        X_mag = numpy.abs(X_spec)
        X_phase = numpy.angle(X_spec)

        return X_mag, X_phase
    
    X_mag, X_phase = preprocess(X_spec)

    coef = X_mag.max()
    X_mag_pre = X_mag / coef

    n_frame = X_mag_pre.shape[2]
    pad_l, pad_r, roi_size = make_padding(n_frame,
                                                data['window_size'], model.offset)
    n_window = int(numpy.ceil(n_frame / roi_size))

    X_mag_pad = numpy.pad(
        X_mag_pre, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')

    if(list(model.state_dict().values())[0].dtype==torch.float16):is_half=True
    else:is_half=False
    pred = _execute(X_mag_pad, roi_size, n_window,
                        device, model, aggressiveness,is_half)
    pred = pred[:, :, :n_frame]
    
    if data['tta']:
        pad_l += roi_size // 2
        pad_r += roi_size // 2
        n_window += 1

        X_mag_pad = numpy.pad(
            X_mag_pre, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')

        pred_tta = _execute(X_mag_pad, roi_size, n_window,
                                device, model, aggressiveness,is_half)
        pred_tta = pred_tta[:, :, roi_size // 2:]
        pred_tta = pred_tta[:, :, :n_frame]

        return (pred + pred_tta) * 0.5 * coef, X_mag, numpy.exp(1.j * X_phase)
    else:
        return pred * coef, X_mag, numpy.exp(1.j * X_phase)

def mask_silence(mag, ref, thres=0.2, min_range=64, fade_size=32):
    if min_range < fade_size * 2:
        raise ValueError('min_range must be >= fade_area * 2')

    mag = mag.copy()

    idx = numpy.where(ref.mean(axis=(0, 1)) < thres)[0]
    starts = numpy.insert(idx[numpy.where(numpy.diff(idx) != 1)[0] + 1], 0, idx[0])
    ends = numpy.append(idx[numpy.where(numpy.diff(idx) != 1)[0]], idx[-1])
    uninformative = numpy.where(ends - starts > min_range)[0]
    if len(uninformative) > 0:
        starts = starts[uninformative]
        ends = ends[uninformative]
        old_e = None
        for s, e in zip(starts, ends):
            if old_e is not None and s - old_e < fade_size:
                s = old_e - fade_size * 2

            if s != 0:
                weight = numpy.linspace(0, 1, fade_size)
                mag[:, :, s:s + fade_size] += weight * ref[:, :, s:s + fade_size]
            else:
                s -= fade_size

            if e != mag.shape[2]:
                weight = numpy.linspace(1, 0, fade_size)
                mag[:, :, e - fade_size:e] += weight * ref[:, :, e - fade_size:e]
            else:
                e += fade_size

            mag[:, :, s + fade_size:e - fade_size] += ref[:, :, s + fade_size:e - fade_size]
            old_e = e

    return mag


class Engine:

    def __init__(self, checkpoint: str, device: str, half: bool, storage: str) -> None:
        self.checkpoint = checkpoint
        self.device = device
        self.half = half
        self.storage = storage
        return
    
    def loadModel(self) -> bool:
        model = InstrumentalExtraction.architecture.CascadedASPPNet(parameter['bins'] * 2)
        archive = torch.load(self.checkpoint, map_location='cpu')
        model.load_state_dict(archive)
        model.eval()
        if(self.half==True): 
            model = model.half().to(self.device)
            pass
        else:
            model = model.to(self.device)
            pass
        self.model = model
        return(True)

    def inferVoice(self, path: str) -> bool:

        data = {
            'postprocess': False,
            'tta': False,
            'window_size': 512,
            'agg': 10,
            'high_end_process': 'mirroring',
        }

        X_wave, X_spec_s = {}, {}
        bands_n = len(parameter['band'])
        for d in range(bands_n, 0, -1): 
            bp = parameter['band'][d]
            if d == bands_n: # high-end band
                X_wave[d], _ = librosa.load(
                    path, sr=bp['sr'], mono=False, dtype=numpy.float32, res_type=bp['res_type'])
                if X_wave[d].ndim == 1:
                    X_wave[d] = numpy.asfortranarray([X_wave[d], X_wave[d]])
            else: # lower bands
                X_wave[d] = librosa.core.resample(X_wave[d+1], orig_sr=parameter['band'][d+1]['sr'], target_sr=bp['sr'], res_type=bp['res_type'])
            # Stft of wave source
            X_spec_s[d] = wave_to_spectrogram_mt(X_wave[d], bp['hl'], bp['n_fft'], parameter['mid_side'], parameter['mid_side_b2'], parameter['reverse'])
            # pdb.set_trace()
            if d == bands_n and data['high_end_process'] != 'none':
                input_high_end_h = (bp['n_fft']//2 - bp['crop_stop']) + ( parameter['pre_filter_stop'] - parameter['pre_filter_start'])
                input_high_end = X_spec_s[d][:, bp['n_fft']//2-input_high_end_h:bp['n_fft']//2, :]

        X_spec_m = combine_spectrograms(X_spec_s)
        aggresive_set = float(data['agg']/100)
        aggressiveness = {'value': aggresive_set, 'split_bin': parameter['band'][1]['crop_stop']}
        with torch.no_grad():
            pred, X_mag, X_phase = inference(X_spec_m,self.device,self.model, aggressiveness, data)
        # Postprocess
        if data['postprocess']:
            pred_inv = numpy.clip(X_mag - pred, 0, numpy.inf)
            pred = mask_silence(pred, pred_inv)
        y_spec_m = pred * X_phase
        v_spec_m = X_spec_m - y_spec_m
        # path = 'a.a.a...a.wav'
        # path[:path.rfind('.')]
        name = os.path.basename(path[:path.rfind('.')])
        folder = os.path.join(self.storage, name)
        os.makedirs(folder, exist_ok=True)
        if ('noise'):
            if data['high_end_process'].startswith('mirroring'):
                input_high_end_ = mirroring(data['high_end_process'], y_spec_m, input_high_end)
                wav_instrument = cmb_spectrogram_to_wave(y_spec_m, input_high_end_h, input_high_end_)
            else:
                wav_instrument = cmb_spectrogram_to_wave(y_spec_m)
            # print ('%s instruments done'%name)
            scipy.io.wavfile.write(os.path.join(folder, 'instruments.wav'), parameter['sr'], (numpy.array(wav_instrument)*32768).astype("int16"))  #
        if ('human'):
            if data['high_end_process'].startswith('mirroring'):
                input_high_end_ = mirroring(data['high_end_process'],  v_spec_m, input_high_end)
                wav_vocals = cmb_spectrogram_to_wave(v_spec_m, input_high_end_h, input_high_end_)
            else:
                wav_vocals = cmb_spectrogram_to_wave(v_spec_m)
            # print ('%s vocals done'%name)
            scipy.io.wavfile.write(os.path.join(folder, 'vocal.wav'), parameter['sr'], (numpy.array(wav_vocals)*32768).astype("int16"))

        return(True)

    pass


def mirroring(a, spec_m, input_high_end):
    if 'mirroring' == a:
        mirror = numpy.flip(numpy.abs(spec_m[:, parameter['pre_filter_start']-10-input_high_end.shape[1]:parameter['pre_filter_start']-10, :]), 1)
        mirror = mirror * numpy.exp(1.j * numpy.angle(input_high_end))
        
        return numpy.where(numpy.abs(input_high_end) <= numpy.abs(mirror), input_high_end, mirror)
        
    if 'mirroring2' == a:
        mirror = numpy.flip(numpy.abs(spec_m[:, parameter['pre_filter_start']-10-input_high_end.shape[1]:parameter['pre_filter_start']-10, :]), 1)
        mi = numpy.multiply(mirror, input_high_end * 1.7)
        
        return numpy.where(numpy.abs(input_high_end) <= numpy.abs(mi), input_high_end, mi)

def cmb_spectrogram_to_wave(spec_m, extra_bins_h=None, extra_bins=None):
    bands_n = len(parameter['band'])    
    offset = 0

    for d in range(1, bands_n + 1):
        bp = parameter['band'][d]
        spec_s = numpy.ndarray(shape=(2, bp['n_fft'] // 2 + 1, spec_m.shape[2]), dtype=complex)
        h = bp['crop_stop'] - bp['crop_start']
        spec_s[:, bp['crop_start']:bp['crop_stop'], :] = spec_m[:, offset:offset+h, :]
        
        offset += h
        if d == bands_n: # higher
            if extra_bins_h: # if --high_end_process bypass
                max_bin = bp['n_fft'] // 2
                spec_s[:, max_bin-extra_bins_h:max_bin, :] = extra_bins[:, :extra_bins_h, :]
            if bp['hpf_start'] > 0:
                spec_s = fft_hp_filter(spec_s, bp['hpf_start'], bp['hpf_stop'] - 1)
            if bands_n == 1:
                wave = spectrogram_to_wave(spec_s, bp['hl'], parameter['mid_side'], parameter['mid_side_b2'], parameter['reverse'])
            else:
                wave = numpy.add(wave, spectrogram_to_wave(spec_s, bp['hl'], parameter['mid_side'], parameter['mid_side_b2'], parameter['reverse']))
        else:
            sr = parameter['band'][d+1]['sr']
            if d == 1: # lower
                spec_s = fft_lp_filter(spec_s, bp['lpf_start'], bp['lpf_stop'])
                wave = librosa.resample(spectrogram_to_wave(spec_s, bp['hl'], parameter['mid_side'], parameter['mid_side_b2'], parameter['reverse']), orig_sr=bp['sr'], target_sr=sr, res_type="sinc_fastest")
            else: # mid
                spec_s = fft_hp_filter(spec_s, bp['hpf_start'], bp['hpf_stop'] - 1)
                spec_s = fft_lp_filter(spec_s, bp['lpf_start'], bp['lpf_stop'])
                wave2 = numpy.add(wave, spectrogram_to_wave(spec_s, bp['hl'], parameter['mid_side'], parameter['mid_side_b2'], parameter['reverse']))
                # wave = librosa.core.resample(wave2, bp['sr'], sr, res_type="sinc_fastest")
                wave = librosa.resample(wave2, orig_sr=bp['sr'], target_sr=sr,res_type='scipy')
        
    return wave.T


def fft_hp_filter(spec, bin_start, bin_stop):
    g = 1.0
    for b in range(bin_start, bin_stop, -1):
        g -= 1 / (bin_start - bin_stop)
        spec[:, b, :] = g * spec[:, b, :]
    
    spec[:, 0:bin_stop+1, :] *= 0

    return spec


def spectrogram_to_wave(spec, hop_length, mid_side, mid_side_b2, reverse):
    spec_left = numpy.asfortranarray(spec[0])
    spec_right = numpy.asfortranarray(spec[1])

    wave_left = librosa.istft(spec_left, hop_length=hop_length)
    wave_right = librosa.istft(spec_right, hop_length=hop_length)

    if reverse:
        return numpy.asfortranarray([numpy.flip(wave_left), numpy.flip(wave_right)])
    elif mid_side:
        return numpy.asfortranarray([numpy.add(wave_left, wave_right / 2), numpy.subtract(wave_left, wave_right / 2)])
    elif mid_side_b2:
        return numpy.asfortranarray([numpy.add(wave_right / 1.25, .4 * wave_left), numpy.subtract(wave_left / 1.25, .4 * wave_right)])
    else:
        return numpy.asfortranarray([wave_left, wave_right])
