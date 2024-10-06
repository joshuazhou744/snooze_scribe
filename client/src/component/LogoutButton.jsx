import React from 'react'
import { useAuth0 } from '@auth0/auth0-react'

export default function LogoutButton({className }) {

    const {logout} = useAuth0();
  return (
    <button onClick={() => logout({returnTo: window.location.origin})} className={`auth-button ${className}`}>
        Log Out
    </button>
  )
}
