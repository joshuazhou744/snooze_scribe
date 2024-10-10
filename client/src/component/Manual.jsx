import React from 'react'
import { Link } from 'react-router-dom'
import "./Manual.css"

export default function Manual() {
  return (
    <div className='container'>
        <div className="header">
            <Link to={"/"}>
                <button className='home-button'>Home</button>
            </Link>
        </div>

        <div className="user-guide">
            <h2>User Guide</h2>
            Guess how to use it
        </div>
    </div>
  )
}
