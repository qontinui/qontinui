/**
 * Example React component demonstrating visibility state detection
 *
 * This component should produce the following states:
 * 1. Page_default (sidebarOpen=false, showModal=false)
 * 2. Page_sidebarOpen_true (sidebarOpen=true, showModal=false, renders Sidebar)
 * 3. Page_showModal_true (sidebarOpen=false, showModal=true, renders Modal)
 * 4. Page_both_open (sidebarOpen=true, showModal=true, renders both)
 *
 * And transitions:
 * - Page_default → Page_sidebarOpen_true (onClick toggle button)
 * - Page_sidebarOpen_true → Page_default (onClick toggle button)
 * - Page_default → Page_showModal_true (onClick open modal button)
 * - Page_showModal_true → Page_default (onClick close modal button)
 */

import React, { useState } from 'react';

function ExamplePage() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [showModal, setShowModal] = useState(false);
  const [menuExpanded, setMenuExpanded] = useState(false);

  return (
    <div className="page-container">
      <header>
        <button onClick={() => setSidebarOpen(!sidebarOpen)}>
          Toggle Sidebar
        </button>
        <button onClick={() => setShowModal(true)}>
          Open Modal
        </button>
        <button onClick={() => setMenuExpanded(!menuExpanded)}>
          Toggle Menu
        </button>
      </header>

      <main>
        {/* Sidebar - conditional rendering with logical AND */}
        {sidebarOpen && (
          <aside className="sidebar">
            <h3>Sidebar Content</h3>
            <nav>
              <a href="/home">Home</a>
              <a href="/about">About</a>
            </nav>
          </aside>
        )}

        <div className="content">
          <h1>Main Content</h1>

          {/* Dropdown menu - conditional rendering */}
          {menuExpanded && (
            <div className="dropdown-menu">
              <ul>
                <li>Option 1</li>
                <li>Option 2</li>
                <li>Option 3</li>
              </ul>
            </div>
          )}
        </div>
      </main>

      {/* Modal - ternary conditional rendering */}
      {showModal ? (
        <div className="modal-overlay">
          <div className="modal">
            <h2>Modal Title</h2>
            <p>Modal content goes here</p>
            <button onClick={() => setShowModal(false)}>
              Close
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}

export default ExamplePage;
