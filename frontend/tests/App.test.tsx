import {afterEach,describe,it,expect,vi} from 'vitest'
import {render,screen,cleanup} from '@testing-library/react'
import userEvent from '@testing-library/user-event'
vi.mock('../prototypes/map-original',()=>({default:()=> <div>Original map surface</div>}))
vi.mock('../src/map',()=>({default:()=> <div>Research map surface</div>}))
import App from '../src/App'
afterEach(cleanup)
describe('Map navigation',()=>{
 it('opens original map by default and labels the active panel',async()=>{
  render(<App />)
  await screen.findByText('Original map surface')
  expect(screen.getByRole('tab',{name:'Original map'}).getAttribute('aria-selected')).toBe('true')
  expect(screen.getByRole('tabpanel').getAttribute('aria-labelledby')).toBe('map-tab-0')
  expect(screen.queryByText('Research map surface')).toBeNull()
 })
 it('switches both ways with clicks and unmounts the inactive map',async()=>{
  const user=userEvent.setup();render(<App />)
  await screen.findByText('Original map surface')
  await user.click(screen.getByRole('tab',{name:'Research map'}))
  await screen.findByText('Research map surface')
  expect(screen.queryByText('Original map surface')).toBeNull()
  await user.click(screen.getByRole('tab',{name:'Original map'}))
  await screen.findByText('Original map surface')
  expect(screen.queryByText('Research map surface')).toBeNull()
 })
 it('supports arrows, Home and End with focus and selection synchronized',async()=>{
  const user=userEvent.setup();render(<App />)
  const original=screen.getByRole('tab',{name:'Original map'});const research=screen.getByRole('tab',{name:'Research map'})
  original.focus();await user.keyboard('{ArrowRight}')
  expect(document.activeElement).toBe(research);expect(research.tabIndex).toBe(0);expect(original.tabIndex).toBe(-1)
  await user.keyboard('{ArrowRight}');expect(document.activeElement).toBe(original)
  await user.keyboard('{End}');expect(document.activeElement).toBe(research)
  await user.keyboard('{Home}');expect(document.activeElement).toBe(original)
  await user.keyboard('{ArrowLeft}');expect(document.activeElement).toBe(research)
  await screen.findByText('Research map surface')
 })
})
