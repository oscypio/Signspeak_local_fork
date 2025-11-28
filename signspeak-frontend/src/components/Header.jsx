import "../../src/styles/Header.css"

function Header({ onNavigate, currentPage }) {
  const handleNavClick = (e, page) => {
    e.preventDefault()
    if (onNavigate) {
      onNavigate(page)
    }
  }

  return (
    <header className="header">
      <img src="/Logo.png" alt="Icon" className="header-icon" />
      <div className="header-list">
        <ul>
          <li>
            <a
              href="/"
              onClick={(e) => handleNavClick(e, "home")}
              className={currentPage === "home" ? "nav-link active" : "nav-link"}
            >
              Home
            </a>
          </li>
          <li>
            <a
              href="/about"
              onClick={(e) => handleNavClick(e, "about")}
              className={currentPage === "about" ? "nav-link active" : "nav-link"}
            >
              About us
            </a>
          </li>
          <li>
            <a
              href="/contact"
              onClick={(e) => handleNavClick(e, "contact")}
              className={currentPage === "contact" ? "nav-link active" : "nav-link"}
            >
              Contact us
            </a>
          </li>
          <li>
            <a
              href="/faq"
              onClick={(e) => handleNavClick(e, "faq")}
              className={currentPage === "faq" ? "nav-link active" : "nav-link"}
            >
              FAQ
            </a>
          </li>
        </ul>
      </div>
    </header>
  )
}

export default Header