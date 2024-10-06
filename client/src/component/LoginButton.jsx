import React from 'react'
import { useAuth0 } from '@auth0/auth0-react'

export default function LoginButton({className}) {

    const { loginWithRedirect } = useAuth0();

  return <button onClick={() => loginWithRedirect()} className={`auth-button ${className}`}>Log In</button>
}
