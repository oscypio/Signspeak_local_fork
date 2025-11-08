import React from 'react'
import '../styles/Header.css'

const Header = () => {
    return (
        <header className='header'>
            <img src="/Logo.png" alt="Icon" className="header-icon" />
            <h1 className='header-title'> SignSpeak</h1>
        </header>
    )
}

export default Header