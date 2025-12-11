import { BiWebcam } from "react-icons/bi"
import { FaHandsAslInterpreting } from "react-icons/fa6"
import { BsFileText } from "react-icons/bs"
import { FaBrain } from "react-icons/fa"
import { GiWorld } from "react-icons/gi"
import { RiUserCommunityFill } from "react-icons/ri"
import { GoArrowRight } from "react-icons/go"
import { FaUserFriends } from "react-icons/fa"
import "../styles/HomePage.css"

function HomePage({ onNavigate }) {
  return (
    <main className="home-page">
      {/* Intro Section */}
      <section className="intro-section">
        <div className="intro-content">
          <div className="logo-container">
            <img src="/Logo.png" alt="SignSpeak Logo" className="intro-logo" />
          </div>
          <p className="intro-subtitle">Bridge the gap between sign and speech</p>
          <button className="translate-button" onClick={() => onNavigate("translate")}>
            Start translating <GoArrowRight />
          </button>
        </div>
      </section>

      {/* How it Works Section */}
      <section className="how-it-works">
        <h2 className="section-title">How it works</h2>
        <div className="features-grid">
          <div className="feature-card">
            <BiWebcam className="feature-icon" />
            <h3>Sign with your webcam</h3>
          </div>
          <div className="feature-card">
            <FaHandsAslInterpreting className="feature-icon" />
            <h3>Accurate sign translate</h3>
          </div>
          <div className="feature-card">
            <FaUserFriends className="feature-icon" />
            <h3>For deaf users</h3>
          </div>
          <div className="feature-card">
            <BsFileText className="feature-icon" />
            <h3>Text</h3>
          </div>
        </div>
      </section>

      {/* Getting Started Section*/}
      <section className="getting-started">
        <h2 className="section-title">Getting Started with SignSpeak</h2>
        <h4>Follow these simple steps to begin translating sign language using your webcam:</h4>
        <div className="steps-container">
          <div className="step">
            <div className="step-number">1</div>
            <div>
              <h3>Choose Your Mode</h3>
              <p>For video calls, generate a Meeting ID and share it with your colleagues so they can read subtitles on their devices.
                  Just want to talk face-to-face? Skip the setup!
                  Simply click Start to use Local Mode instantly for in-person conversations without any internet configuration.</p>
            </div>
          </div>
          <div className="step">
            <div className="step-number">2</div>
            <div>
              <h3>Launch Picture-in-Picture</h3>
              <p>
                  Don't hide your video call or your other apps! Click "Setup for meeting" and select "Launch PiP".
                  This detaches the translation window, allowing you to float the subtitles right on top of Zoom, Google Meet, or Teams, so you never lose eye contact.
              </p>
            </div>
          </div>
          <div className="step">
            <div className="step-number">3</div>
            <div>
              <h3>Sign & Correct</h3>
              <p>
                  Start signing slowly and clearly. The AI will show partial words in gray as you move, giving you instant feedback.
                  Once you finish a sentence, it automatically refines it into fluent English.
                  Made a mistake? Just hit the Restart Icon to instantly clear the buffer and start the phrase over.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Trust Section */}
      <section className="trust-section">
        <h2 className="trust-title">The Sign Language AI trusted by inclusive communities.</h2>
        <p className="trust-subtitle">
          First, join your video call on any platform (like Google Meet, Zoom, or Teams).
          SignSpeak is your smart assistant for real-time communication between spoken language and sign language.
          Enjoy seamless and accessible interactions during video calls — all while keeping your data safe and private.
        </p>
        <div className="trust-features">
          <div className="trust-feature">
            <FaBrain className="trust-icon" />
            <h3>AI-powered Sign Language recognition</h3>
          </div>
          <div className="trust-feature">
            <RiUserCommunityFill className="trust-icon" />
            <h3>Designed for accessibility & inclusion</h3>
          </div>
          <div className="trust-feature">
            <GiWorld className="trust-icon" />
            <h3>Built for the Deaf community</h3>
          </div>
        </div>
      </section>
    </main>
  )
}

export default HomePage