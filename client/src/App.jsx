import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Auth0ProviderWithHistory from './auth/Auth0Provider';
import AudioRecorder from './component/AudioRecorder';
import Callback from './component/Callback';
import Manual from './component/Manual';

function App() {
  return (
    <Router>
      <Auth0ProviderWithHistory>
        <Routes>
          <Route path="/" element={<AudioRecorder />} />
          <Route path="/callback" element={<Callback />} />
          <Route path="/user-guide" element={<Manual />} />
        </Routes>
      </Auth0ProviderWithHistory>
    </Router>
  );
}

export default App;