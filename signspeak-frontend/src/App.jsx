import React, { useState } from 'react';
import HomePage from './pages/HomePage';
import Header from './components/Header'
import TranslationPage from './pages/TranslationPage';
import './styles/index.css';
import Footer from './components/footer';

function App() {
    const [currentPage, setCurrentPage] = useState('home');

    const renderPage = () => {
        switch (currentPage) {
            case 'home':
                return <HomePage onNavigate={handleNavigate} />;
            case 'translate':
                return <TranslationPage onNavigate={handleNavigate} />;
            default:
                return <HomePage />;
        }
    };

    const handleNavigate = (pageName) => {
        setCurrentPage(pageName);
    };

    return (
        <div className="app">
            <Header />
            <main className="main-content">
                {renderPage()}
            </main>
            <Footer />
        </div>
    );
}

export default App;

