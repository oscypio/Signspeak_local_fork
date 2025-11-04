import React from 'react';

function HomePage({ onNavigate }) {
    return (
        <main className="main-content">
            <div style={{ padding: '2rem', textAlign: 'center' }}>
                <p>This is the home page. Click "Get Start" to begin translation.</p>
            </div>
            <button
                onClick={() => onNavigate('translate')}
            >
                Get Start
            </button>
        </main>
    );
}

export default HomePage;
