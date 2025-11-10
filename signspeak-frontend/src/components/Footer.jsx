import React from 'react'
import '../styles/Footer.css'
import { FaFacebook, FaLinkedin, FaYoutube, FaInstagram } from "react-icons/fa";


const Footer = () => {
    
  return (
    <footer className="footer">
      <div className="footer-container">
        <div className="footer-content">
          <div className="footer-section">
            <h4>SignSpeak</h4>
            <div className="social-links">
              <a href="#"><FaFacebook /></a>
              <a href="#"><FaLinkedin /></a>
              <a href="#"><FaYoutube /></a>
              <a href="#"><FaInstagram /></a>
            </div>
          </div>
          <div className="footer-section">
            <h4>Product</h4>
            <ul>
              <li>
                <a href="/">Home</a>
              </li>
              <li>
                <a href="/FAQ">FAQ</a>
              </li>
            </ul>
          </div>
          <div className="footer-section">
            <h4>Company</h4>
            <ul>
              <li>
                <a href="/About">About us</a>
              </li>
              <li>
                <a href="/Contact">Contact</a>
              </li>
            </ul>
          </div>
        </div>
        <div className="footer-bottom">
          <p>&copy; 2025 SignSpeak. All rights reserved.</p>
        </div>
      </div>
    </footer>
  )
}

export default Footer