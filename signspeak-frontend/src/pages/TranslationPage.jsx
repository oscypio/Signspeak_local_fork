import "../styles/TranslationPage.css";
import { useTranslation } from '../context/TranslationContext';
import ControlPanel from '../components/ControlPanel';
import VideoDisplay from '../components/VideoDisplay';
import { HiMiniSpeakerWave } from "react-icons/hi2";
import { MdOutlineTextIncrease, MdOutlineTextDecrease } from "react-icons/md";
import { FiSettings } from "react-icons/fi";
import { VscDebugRestart } from "react-icons/vsc";

function TranslationPage() {
    const {
        mediaStream, isWebcamOn, status, toggleWebcam,
        translatedText, partialWords, readAloud,
        fontSize, increaseFont, decreaseFont,
        videoRef, pipWindow,
        setShowConfigModal,
        restartTranslation
    } = useTranslation();


    return (
        <main className="translation-content">

            <section className="video-section">
                <VideoDisplay ref={videoRef} status={status} stream={mediaStream} />
                <div className="buttons-container">
                    <ControlPanel onClick={toggleWebcam} isWebcamOn={isWebcamOn} status={status} />
                    <button className="setup-pip-btn" onClick={() => setShowConfigModal(true)}>
                        <FiSettings style={{ marginRight: '8px' }} /> {pipWindow ? 'PiP Active' : 'Setup for meeting'}
                    </button>
                </div>
            </section>

            <section className="translation-section">
                <div className="translation-box">
                    <div className="result-text" style={{ fontSize: `${fontSize}px` }}>
                        {translatedText && <span style={{ color: '#0d2538', display: 'block', marginBottom: '10px' }}>{translatedText}</span>}
                        {partialWords.length > 0 && <span style={{ color: '#888', fontStyle: 'italic' }}>{partialWords.join(' ')}...</span>}
                        {!translatedText && partialWords.length === 0 && <span style={{ color: '#ccc' }}>Waiting...</span>}
                    </div>
                    <div className="translation-actions">
                        <button className="translated-text" onClick={readAloud}><HiMiniSpeakerWave /></button>
                        <button className="increase-font" onClick={increaseFont}><MdOutlineTextIncrease /></button>
                        <button className="decrease-font" onClick={decreaseFont}><MdOutlineTextDecrease /></button>
                        <button className="restart-btn" onClick={restartTranslation} title="Clear Context">
                            <VscDebugRestart />
                        </button>
                    </div>
                </div>
            </section>

        </main>
    );
}

export default TranslationPage;