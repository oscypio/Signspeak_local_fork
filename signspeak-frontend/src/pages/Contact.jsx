import { useState } from "react"
import { MdEmail, MdLocationOn, MdPhone } from "react-icons/md"
import { FaTwitter, FaLinkedin, FaGithub } from "react-icons/fa"
import "../styles/Contact.css"

function Contact({onNavigate}) {
    
    const [formData, setFormData] = useState({
        name: "",
        lastName: "",
        email: "",
        phone: "",
        subject: "",
        message: "",
    })

    const [submitStatus, setSubmitStatus] = useState(null)

    const handleChange = (e) => {
        const { name, value } = e.target
        setFormData((prev) => ({
        ...prev,
        [name]: value,
        }))
    }

    const handleSubmit = (e) => {
        e.preventDefault()
        console.log("Form submitted:", formData)
        setSubmitStatus("success")

        // Reset form after submission
        setTimeout(() => {
        setFormData({
            name: "",
            lastName: "",
            email: "",
            phone: "",
            subject: "",
            message: "",
        })
        setSubmitStatus(null)
        }, 3000)
    }

    return (
        <main className="contact-page">
            <section className="contact-section">
                <div>
                    <div className="text-section">
                        <h1>Contact us</h1>
                        <p className="contact-text">
                            Contact us for support, feedback, or any questions about SignSpeak.
                            <br /><br />
                            If you would like to report an issue, learn more about our technology,
                            or help us improve communication accessibility. We would love to hear from you.
                        </p>
                    </div>

                    <div className="form-section">
                        <form onSubmit={handleSubmit} className="contact-form">
                            <div className="form-row">
                                <div className="form-group">
                                    <label htmlFor="name">Name</label>
                                    <input
                                        type="text"
                                        id="name"
                                        name="name"
                                        value={formData.name}
                                        onChange={handleChange}
                                        placeholder="Enter your name"
                                        required
                                    />
                                </div>
                                <div className="form-group">
                                    <label htmlFor="lastName">Last name</label>
                                    <input
                                        type="text"
                                        id="lastName"
                                        name="lastName"
                                        value={formData.lastName}
                                        onChange={handleChange}
                                        placeholder="Enter your last name"
                                        required
                                    />
                                </div>
                            </div>

                            <div className="form-row">
                                <div className="form-group">
                                    <label htmlFor="email">Email address</label>
                                    <input
                                        type="email"
                                        id="email"
                                        name="email"
                                        value={formData.email}
                                        onChange={handleChange}
                                        placeholder="user@example.com"
                                        required
                                    />
                                </div>
                                <div className="form-group">
                                    <label htmlFor="phone">Contact number</label>
                                    <input
                                        type="tel"
                                        id="phone"
                                        name="phone"
                                        value={formData.phone}
                                        onChange={handleChange}
                                        placeholder="Enter your contact number"
                                    />
                                </div>
                            </div>

                            <div className="form-group full-width">
                                <label htmlFor="message">Message</label>
                                <textarea
                                    id="message"
                                    name="message"
                                    value={formData.message}
                                    onChange={handleChange}
                                    placeholder="Enter your message or request..."
                                    rows={5}
                                    required
                                />
                            </div>

                            <button type="submit" className="submit-btn">Submit</button>
                            {submitStatus === "success" && (
                                <p className="success-message">Message sent successfully!</p>
                            )}
                        </form>
                    </div>
                </div>
            </section>

        </main>
    )
}

export default Contact