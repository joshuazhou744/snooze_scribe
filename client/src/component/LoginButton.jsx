import PropTypes from 'prop-types';
import { useAuth0 } from '@auth0/auth0-react';

function LoginButton({ className = '', isAuthenticated = false }) {

    const { loginWithRedirect } = useAuth0();
    const handleLogin = () => {
      loginWithRedirect({
        redirectUri: `${window.location.origin}/callback`,
      });
    };

  return <button disabled={isAuthenticated} onClick={handleLogin} className={`auth-button ${className}`}>Log In</button>
}

LoginButton.propTypes = {
  className: PropTypes.string,
  isAuthenticated: PropTypes.bool,
};

export default LoginButton;
