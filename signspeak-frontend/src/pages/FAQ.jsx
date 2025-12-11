import { useState } from "react"
import { FaChevronDown, FaChevronUp } from "react-icons/fa"
import "../styles/FAQ.css"

function FAQ() {
    const [openIndex, setOpenIndex] = useState(null)

    const faqs = [
        {
            question: "What is SignSpeak?",
            answer: "SignSpeak is an AI-powered web application that translates American Sign Language (ASL) into text and speech in real-time. It uses a smart Picture-in-Picture overlay to work seamlessly on top of any video calling platform like Zoom, Google Meet, or Teams.",
        },
        {
            question: "How accurate is the translation?",
            answer: "SignSpeak uses a two-step process to ensure accuracy. First, it recognizes individual signs in real-time providing instant feedback. Then, once you finish a sentence, a Large Language Model refines the sequence into fluent, grammatically correct English.",
        },
        {
            question: "Which sign languages are supported?",
            answer: "Currently, SignSpeak is specialized in American Sign Language (ASL). We are focused on perfecting the ASL recognition model before expanding to other sign languages like BSL or IS.",
        },
        {
            question: "Do I need to install plugins or special cameras?",
            answer: "No! SignSpeak works directly in your browser (Chrome or Edge recommended) with your standard webcam. There is no need to install plugins or buy special hardware. We use a 'Local Mode' for face-to-face chat and a 'Meeting ID' system for remote syncing.",
        },
        {
            question: "How does the 'Picture-in-Picture' mode work?",
            answer: "This is our key feature! Instead of sharing your screen, you launch a floating window that stays on top of your other apps. This allows you to see the translation subtitles right next to the person you are talking to on Zoom or Meet, without hiding their video.",
        },
        {
            question: "Does it work on mobile phones?",
            answer: "SignSpeak is optimized for Desktop/Laptop browsers (Chrome, Edge) to leverage the advanced Picture-in-Picture API and processing power required for the AI. While the website is responsive, the translation features work best on a computer.",
        },
        {
            question: "How does SignSpeak protect my privacy?",
            answer: "We take privacy very seriously. The video analysis (hand tracking) happens locally on your device using MediaPipe. We do not stream your raw video to our servers—only the abstract coordinates of your hand movements are sent to generate the translation.",
        },
        {
            question: "What if the system misinterprets a sign?",
            answer: "It happens! That's why we included a 'Restart' button. If you see the real-time feedback is incorrect, you can instantly clear the current sentence and start over to ensure your message is conveyed clearly.",
        },
        {
            question: "Is SignSpeak free?",
            answer: "Yes, SignSpeak is currently a free-to-use tool designed to break down communication barriers and improve accessibility for the Deaf community.",
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