import "../styles/About.css"

function About({onNavigate}) {
    return (
        <main className="about-page">
            {/* Text About us Section */}
            <section className="about-section">
                <div>
                    <div className="text-section">
                        <h1>About</h1>
                        <p className="about-text">
                            SignSpeak was born from a simple idea: communication is a universal human right, but not yet a universal reality. We saw a gap between communities and believed that technology could be the bridge.
                            <br />
                            Our Mission: Our mission is to break down communication barriers between the Deaf and hearing communities. We are dedicated to creating intuitive, AI-powered tools that make conversations in American Sign Language (ASL) accessible to everyone, on any platform, without the need for expensive hardware or human interpreters.
                            <br />
                            How It Works: We use cutting-edge AI and machine learning to analyze ASL gestures directly from your webcam. Our models interpret these dynamic signs in real-time, converting them into spoken or written text. This creates a seamless experience that allows for fluid conversation. We are committed to building a more inclusive and connected world, one sign at a time.
                        </p>
                    </div>
                    <div className="imagine-container">
                        <img src="/aboutUsImage.png" alt="About Us Image" className="About-us-image" />
                    </div>
                </div>
            </section>

            <section className="team-section">
                <div className="team-container">
                    <h2 className="team-title">Our team</h2>
                    <div className="team-grid">
                        <div className="team-card">
                            <div className="team-avatar">
                                <svg viewBox="0 0 24 24" fill="currentColor">
                                    <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>
                                </svg>
                            </div>
                            <h3 className="member-name">Karol Jinet Cardona Bolanos</h3>
                            <p className="member-role">Product Owner • Frontend Developer • Tester</p>
                            <p className="member-description">Drives product vision and ensures seamless UI integration.</p>
                        </div>

                        <div className="team-card">
                            <div className="team-avatar">
                                <svg viewBox="0 0 24 24" fill="currentColor">
                                    <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>
                                </svg>
                            </div>
                            <h3 className="member-name">Elena Martinez Vazquez</h3>
                            <p className="member-role">UX/UI Designer & Frontend Developer</p>
                            <p className="member-description">Creates accessible and intuitive user experiences.</p>
                        </div>

                        <div className="team-card">
                            <div className="team-avatar">
                                <svg viewBox="0 0 24 24" fill="currentColor">
                                    <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>
                                </svg>
                            </div>
                            <h3 className="member-name">Michal Aleksander Sachanbinski</h3>
                            <p className="member-role">Machine Learning Engineer</p>
                            <p className="member-description">Designs and trains the core sign-recognition neural networks.</p>
                        </div>

                        <div className="team-card">
                            <div className="team-avatar">
                                <svg viewBox="0 0 24 24" fill="currentColor">
                                    <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>
                                </svg>
                            </div>
                            <h3 className="member-name">Mohamed Salah Abdelaziz Mohamed Heikal</h3>
                            <p className="member-role">Product Manager & ML Engineer</p>
                            <p className="member-description">Coordinates the ML pipeline and business/technical alignment.</p>
                        </div>

                        <div className="team-card">
                            <div className="team-avatar">
                                <svg viewBox="0 0 24 24" fill="currentColor">
                                    <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>
                                </svg>
                            </div>
                            <h3 className="member-name">Kuessin-Ansan Manuel DJIYEHOUE</h3>
                            <p className="member-role">Backend Developer</p>
                            <p className="member-description">Builds APIs, WebSockets and ensures real-time communication.</p>
                        </div>
                    </div>
                </div>
            </section>
        </main>
    )
}

export default About