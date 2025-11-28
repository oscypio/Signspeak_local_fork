import React, { useEffect, useRef } from 'react';
import { HiMiniSpeakerWave } from "react-icons/hi2";
import { MdOutlineTextIncrease, MdOutlineTextDecrease } from "react-icons/md";

const PiPContent = ({
                        isDeaf,
                        stream,
                        isWebcamOn,
                        toggleWebcam,
                        translatedText,
                        partialWords,
                        fontSize,
                        increaseFont,
                        decreaseFont,
                        readAloud
                    }) => {
    const videoRef = useRef(null);

    // Handle video stream for deaf users
    useEffect(() => {
        if (isDeaf && stream && videoRef.current) {
            videoRef.current.srcObject = stream;
            videoRef.current.play().catch(e => console.log("PiP play error", e));
        }
    }, [stream]);

    return (
        <div className="pip-wrapper" style={{ display: 'flex', flexDirection: 'column', height: '100%', padding: '20px', boxSizing: 'border-box' }}>

            {/* Header */}
            <div className="pip-header" style={{ marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '10px' }}>
                <img src="/Logo.png" alt="SignSpeak" style={{ height: '30px' }} />
            </div>

            <div className="video-section" style={{
                flex: '0 0 auto',
                width: '100%',
                marginBottom: isDeaf ? '20px' : '0' ,
                display: isDeaf ? 'block' : 'none'
            }}>
                <video
                    ref={videoRef}
                    style={{ width: '100%', borderRadius: '8px', transform: 'scaleX(-1)', background: 'black' }}
                    autoPlay
                    muted
                    playsInline
                />
            </div>

            {!isDeaf && (
                <div style={{ position: 'absolute', opacity: 0, pointerEvents: 'none', width: '1px', height: '1px', overflow: 'hidden' }}>
                    <video
                        ref={videoRef}
                        autoPlay
                        muted
                        playsInline
                    />
                </div>
            )}

            {/* Video Section (Solo Deaf) */}
            {isDeaf ? (
                <button className="start-btn" onClick={toggleWebcam} style={{ width: '100%', marginBottom: '20px' }}>
                    {isWebcamOn ? "Stop" : "Start"}
                </button>

            ) : (
                <div style={{ textAlign: 'center', marginBottom: '20px', color: '#666', fontWeight: '500' }}>
                    Listening Mode Active
                </div>
            )}

            {/* Translation Box */}
            <div className="translation-box" style={{ flex: '1', display: 'flex', flexDirection: 'column' }}>
                <div className="result-text" style={{ fontSize: `${fontSize}px`, flex: '1', overflowY: 'auto' }}>

                    {/* Testo Finale */}
                    {translatedText && (
                        <span style={{ color: '#0d2538', display: 'block', marginBottom: '5px' }}>
                            {translatedText}
                        </span>
                    )}

                    {/* Partial text (Real-time) */}
                    {partialWords.length > 0 && (
                        <span style={{ color: '#888', fontStyle: 'italic' }}>
                            {partialWords.join(' ')}...
                        </span>
                    )}

                    {/* Placeholder */}
                    {!translatedText && partialWords.length === 0 && (
                        <span style={{ color: '#ccc' }}>Waiting...</span>
                    )}
                </div>

                <div className="translation-actions" style={{ marginTop: '10px' }}>
                    <button className="translated-text" onClick={readAloud}><HiMiniSpeakerWave /></button>
                    <button className="increase-font" onClick={increaseFont}><MdOutlineTextIncrease /></button>
                    <button className="decrease-font" onClick={decreaseFont}><MdOutlineTextDecrease /></button>
                </div>
            </div>
        </div>
    );
};

export default PiPContent;