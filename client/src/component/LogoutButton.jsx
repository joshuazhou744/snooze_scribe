import PropTypes from 'prop-types';
import { useAuth0 } from '@auth0/auth0-react';

function LogoutButton({ className = '' }) {

    const {logout} = useAuth0();

    const handleLogout = () => {
      logout({
        returnTo: window.location.origin, 
        federated: true
      })
      localStorage.clear()
    }
  return (
    <button onClick={handleLogout} className={`auth-button ${className}`}>
        Log Out
    </button>
  )
}

LogoutButton.propTypes = {
  className: PropTypes.string,
};

export default LogoutButton;
