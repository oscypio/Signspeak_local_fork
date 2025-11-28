"use client"

import { useState } from "react"
import HomePage from "./pages/HomePage"
import Header from "./components/Header"
import TranslationPage from "./pages/TranslationPage"
import About from "./pages/About"
import Contact from "./pages/Contact"
import FAQ from "./pages/FAQ"
import "./styles/index.css"
import Footer from "./components/Footer"
import { TranslationProvider } from './context/TranslationContext';

function App() {
  const [currentPage, setCurrentPage] = useState("home")

  const renderPage = () => {
    switch (currentPage) {
      case "home":
        return <HomePage onNavigate={handleNavigate} />
      case "translate":
        return <TranslationPage onNavigate={handleNavigate} />
      case "about":
        return <About />
      case "contact":
        return <Contact />
      case "faq":
        return <FAQ />
      default:
        return <HomePage onNavigate={handleNavigate} />
    }
  }

  const handleNavigate = (pageName) => {
    setCurrentPage(pageName)
  }

  return (
      <TranslationProvider>
        <div className="app">
          <Header onNavigate={handleNavigate} currentPage={currentPage} />
          <main className="main-content">{renderPage()}</main>
          <Footer />
        </div>
      </TranslationProvider>
  )
}

export default App