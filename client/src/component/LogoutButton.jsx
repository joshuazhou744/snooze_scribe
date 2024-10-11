import React from 'react'
import { useAuth0 } from '@auth0/auth0-react'

export default function LogoutButton({className }) {

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
