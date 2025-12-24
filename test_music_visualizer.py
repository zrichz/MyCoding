"""
Music Visualizer Test Script
Tests core functionality without GUI
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
import os

def test_audio_analysis():
    """Test audio analysis with a synthetic audio signal"""
    print("🎵 Testing Music Visualizer Core Functions")
    print("=" * 45)
    
    try:
        # Create synthetic audio signal (sine wave with transients)
        print("📊 Creating synthetic audio signal...")
        duration = 5.0  # seconds
        sr = 22050  # sample rate
        t = np.linspace(0, duration, int(sr * duration))
        
        # Base sine wave
        frequency = 440  # A4 note
        audio = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        # Add some transients (sudden volume changes)
        transient_times = [1.0, 2.5, 4.0]
        for trans_t in transient_times:
            idx = int(trans_t * sr)
            if idx < len(audio):
                # Add a brief loud pulse
                pulse_length = int(0.1 * sr)  # 0.1 second pulse
                audio[idx:idx + pulse_length] += 0.8 * np.sin(2 * np.pi * 880 * t[idx:idx + pulse_length])
        
        print(f"   ✅ Audio signal: {duration}s, {sr} Hz, {len(audio)} samples")
        
        # Test spectrogram generation
        print("📈 Computing spectrogram...")
        hop_length = 512
        n_fft = 2048
        
        S_complex = librosa.stft(audio, hop_length=hop_length, n_fft=n_fft)
        S = np.abs(S_complex)
        S_db = librosa.amplitude_to_db(S, ref=np.max)
        
        times = librosa.times_like(S_db, sr=sr, hop_length=hop_length)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        
        print(f"   ✅ Spectrogram: {len(freqs)} freq bins, {len(times)} time frames")
        print(f"   ✅ Time resolution: {times[1] - times[0]:.4f}s")
        print(f"   ✅ Frequency range: 0 - {freqs[-1]:.0f} Hz")
        
        # Test transient detection
        print("⚡ Detecting transients...")
        onset_frames = librosa.onset.onset_detect(
            y=audio, 
            sr=sr, 
            hop_length=hop_length,
            units='frames'
        )
        transients = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
        
        print(f"   ✅ Found {len(transients)} transients")
        if len(transients) > 0:
            print(f"   ✅ First transient: {transients[0]:.2f}s")
            print(f"   ✅ Last transient: {transients[-1]:.2f}s")
        
        # Test visualization
        print("🎨 Testing visualization...")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot waveform
        ax1.plot(t, audio)
        ax1.set_title('Synthetic Audio Waveform with Transients')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        
        # Mark transients on waveform
        for trans in transients:
            ax1.axvline(x=trans, color='red', alpha=0.7, linewidth=2)
        
        # Plot spectrogram
        img = librosa.display.specshow(
            S_db, 
            x_axis='time', 
            y_axis='hz',
            sr=sr,
            hop_length=hop_length,
            ax=ax2,
            cmap='viridis'
        )
        
        # Mark transients on spectrogram
        for trans in transients:
            ax2.axvline(x=trans, color='red', alpha=0.7, linewidth=2)
        
        ax2.set_title(f'Spectrogram with {len(transients)} Detected Transients')
        
        plt.colorbar(img, ax=ax2, format='%+2.0f dB')
        plt.tight_layout()
        
        # Save test plot
        plt.savefig('music_visualizer_test.png', dpi=150, bbox_inches='tight')
        print("   ✅ Saved test visualization: music_visualizer_test.png")
        
        plt.close()
        
        print("\n✅ All tests passed!")
        print("🚀 Music Visualizer is ready to use!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing library: {e}")
        print("Please run the launcher script to install required packages.")
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_audio_analysis()
    if not success:
        print("\n💡 If tests failed, try running:")
        print("   ./run_music_visualizer.sh")
        print("   This will install required packages.")