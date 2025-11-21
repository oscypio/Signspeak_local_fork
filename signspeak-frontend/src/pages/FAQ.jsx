import { useState } from "react"
import { FaChevronDown, FaChevronUp } from "react-icons/fa"
import "../styles/FAQ.css"

function FAQ() {
    const [openIndex, setOpenIndex] = useState(null)

    const faqs = [
        {
            question: "What is SignSpeak?",
            answer: "SignSpeak is an AI-powered platform that translates sign language gestures into text and speech in real-time. It uses advanced computer vision and machine learning to recognize hand movements and convert them into understandable communication.",
        },
        {
            question: "How accurate is the sign language recognition?",
            answer: "Our AI model achieves over 95% accuracy in recognizing common sign language gestures. The accuracy improves with clear lighting, proper hand positioning, and consistent signing patterns. We continuously update our models to improve recognition accuracy.",
        },
        {
            question: "Which sign languages are supported?",
            answer: "Currently, SignSpeak supports American Sign Language (ASL), British Sign Language (BSL), and International Sign. We're actively working on adding support for more regional sign languages including ASL variations used in different countries.",
        },
        {
            question: "Do I need special equipment to use SignSpeak?",
            answer: "No special equipment is required! SignSpeak works with any standard webcam or device camera. For best results, we recommend good lighting and a clear background behind the signer.",
        },
        {
            question: "Is SignSpeak free to use?",
            answer: "Yes, SignSpeak offers a free tier with basic translation features. We also offer premium plans for businesses and educational institutions that include advanced features, higher usage limits, and priority support.",
        },
        {
            question: "Can SignSpeak work in real-time conversations?",
            answer: "Yes! SignSpeak processes gestures in real-time with minimal latency (typically under 200ms). This makes it suitable for live conversations, video calls, and interactive communication.",
        },
        {
            question: "How does SignSpeak protect my privacy?",
            answer: "We take privacy seriously. Video data is processed locally on your device whenever possible, and we never store your video recordings without explicit permission. All data transmission is encrypted, and we comply with GDPR and other privacy regulations.",
        },
        {
            question: "Can I use SignSpeak on mobile devices?",
            answer: "Yes, SignSpeak is fully responsive and works on smartphones and tablets. We also offer dedicated mobile apps for iOS and Android with optimized performance for mobile cameras.",
        },
        {
            question: "What if the system doesn't recognize my signs?",
            answer: "If recognition issues occur, try adjusting your lighting, ensuring your hands are fully visible, or signing more slowly. You can also report unrecognized signs to help us improve our AI models. Our support team is always available to help troubleshoot.",
        },
        {
            question: "How can I contribute to improving SignSpeak?",
            answer: "We welcome community contributions! You can help by providing feedback, reporting recognition errors, participating in our beta testing program, or contributing to our open-source gesture dataset. Contact us for more information about getting involved.",
        },
    ]

    const toggleFAQ = (index) => {
        setOpenIndex(openIndex === index ? null : index)
    }

    return (
        <main className="faq-page">
            {/* Intro Section */}
            <section className="faq-intro">
                <h2>We thought they would be useful for you too.</h2>
                <h1>Our support team answers these<br />questions almost daily.</h1>
            </section>

            {/* FAQ Section */}
            <section className="faq-content">
                <div className="faq-container">
                    {faqs.map((faq, index) => (
                    <div key={index} className={`faq-item ${openIndex === index ? "active" : ""}`}>
                        <button className="faq-question" onClick={() => toggleFAQ(index)} aria-expanded={openIndex === index}>
                            <span>{faq.question}</span>
                            {openIndex === index ? <FaChevronUp /> : <FaChevronDown />}
                        </button>
                        {openIndex === index && (
                        <div className="faq-answer">
                            <p>{faq.answer}</p>
                        </div>
                        )}
                    </div>
                    ))}
                </div>
            </section>
        </main>
    )
}

export default FAQ