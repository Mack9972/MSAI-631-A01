import React, { useMemo, useState } from 'react';

function App() {
  const [highContrast, setHighContrast] = useState(false);
  const [largeText, setLargeText] = useState(false);
  const [reduceMotion, setReduceMotion] = useState(false);
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [announcement, setAnnouncement] = useState('');

  const themeClass = useMemo(() => {
    const classes = ['app'];
    if (highContrast) classes.push('theme-contrast');
    if (largeText) classes.push('theme-large-text');
    if (reduceMotion) classes.push('theme-reduced-motion');
    return classes.join(' ');
  }, [highContrast, largeText, reduceMotion]);

  function handleSubmit(event) {
    event.preventDefault();
    setAnnouncement(
      `Thanks ${name || 'guest'}! Your feedback was submitted successfully.`
    );
  }

  return (
    <div className={themeClass}>
      <header className="hero" role="banner">
        <p className="eyebrow">React Adaptive Accessibility UI</p>
        <h1>Accessibility Guidelines, Demonstrated Top to Bottom</h1>
        <p>
          This interface is organized in WCAG POUR order: Perceivable, Operable,
          Understandable, and Robust.
        </p>
      </header>

      <main id="main-content" tabIndex="-1" className="stack-layout">
        <section className="panel controls-panel" aria-labelledby="controls-title">
          <h2 id="controls-title">Adaptive Controls</h2>
          <p>Use these controls to see the interface adapt in real time.</p>
          <div className="toggle-grid" role="group" aria-label="Accessibility settings">
            <label className="toggle-card" htmlFor="highContrast">
              <input
                id="highContrast"
                type="checkbox"
                checked={highContrast}
                onChange={(e) => setHighContrast(e.target.checked)}
              />
              <span>High Contrast</span>
            </label>

            <label className="toggle-card" htmlFor="largeText">
              <input
                id="largeText"
                type="checkbox"
                checked={largeText}
                onChange={(e) => setLargeText(e.target.checked)}
              />
              <span>Larger Text</span>
            </label>

            <label className="toggle-card" htmlFor="reduceMotion">
              <input
                id="reduceMotion"
                type="checkbox"
                checked={reduceMotion}
                onChange={(e) => setReduceMotion(e.target.checked)}
              />
              <span>Reduced Motion</span>
            </label>
          </div>
        </section>

        <section className="panel guideline" aria-labelledby="perceivable-title">
          <h2 id="perceivable-title">1. Perceivable</h2>
          <p>
            Information is available through multiple sensory channels with readable
            contrast and text alternatives.
          </p>
          <figure>
            <img
              src="/accessibility-illustration.svg"
              alt="Three people collaborating around a table with laptops and notes"
            />
            <figcaption>
              The image includes descriptive alternative text for screen reader users.
            </figcaption>
          </figure>
        </section>

        <section className="panel guideline" aria-labelledby="operable-title">
          <h2 id="operable-title">2. Operable</h2>
          <p>
            Controls are keyboard accessible, clearly focused, and predictable in
            behavior.
          </p>
          <div className="button-row" role="group" aria-label="Keyboard operable actions">
            <button type="button" className="fancy-btn">Primary Action</button>
            <button type="button" className="fancy-btn fancy-btn-secondary">Secondary Action</button>
            <button type="button" className="fancy-btn fancy-btn-ghost">Tertiary Action</button>
          </div>
        </section>

        <section className="panel guideline" aria-labelledby="understandable-title">
          <h2 id="understandable-title">3. Understandable</h2>
          <p>
            Form labels, helper text, and consistent interaction reduce confusion and
            cognitive load.
          </p>
          <form onSubmit={handleSubmit} noValidate>
            <div className="field">
              <label htmlFor="name">Name</label>
              <input
                id="name"
                name="name"
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                autoComplete="name"
                placeholder="Enter your name"
              />
            </div>

            <div className="field">
              <label htmlFor="email">Email</label>
              <input
                id="email"
                name="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                autoComplete="email"
                aria-describedby="email-help"
                placeholder="you@example.com"
              />
              <small id="email-help">Only used for class project follow-up.</small>
            </div>

            <button type="submit" className="fancy-btn">Submit Feedback</button>
          </form>
        </section>

        <section className="panel guideline" aria-labelledby="robust-title">
          <h2 id="robust-title">4. Robust</h2>
          <p>
            Semantic structure and ARIA live regions support assistive technology
            compatibility.
          </p>
          <p className="sr-announcement" aria-live="polite" aria-atomic="true">
            {announcement || 'Status updates appear here after submitting the form.'}
          </p>
        </section>
      </main>
    </div>
  );
}

export default App;
