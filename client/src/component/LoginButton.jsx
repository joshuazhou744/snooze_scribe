import React from 'react'
import { useAuth0 } from '@auth0/auth0-react'

export default function LoginButton({className, isAuthenticated}) {

    const { loginWithRedirect } = useAuth0();
    const handleLogin = () => {
      loginWithRedirect({
        redirectUri: `${window.location.origin}/callback`,
      });
    };

  return <button disabled={isAuthenticated} onClick={handleLogin} className={`auth-button ${className}`}>Log In</button>
}
