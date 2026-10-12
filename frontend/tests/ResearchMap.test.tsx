import {afterEach,it,expect} from 'vitest'
import {cleanup,render,screen,within} from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import ResearchMap from '../src/map'
import data from '../src/research-data.json'
afterEach(cleanup)
it('shows the verified historical totals and updates to the selected year',async()=>{
 const user=userEvent.setup();render(<ResearchMap />)
 const summary=screen.getByRole('region',{name:'Selected period summary'})
 expect(within(summary).getByText('140,806')).toBeTruthy()
 expect(within(summary).getByText('2,475')).toBeTruthy()
 await user.selectOptions(screen.getByLabelText('Period'),'2024')
 expect(within(summary).getByText('33,332')).toBeTruthy()
 expect(within(summary).getByText('537')).toBeTruthy()
})
it('does not retain stale area statistics when a selected cell has no records in the new year',async()=>{
 const user=userEvent.setup();render(<ResearchMap />)
 const cell=data.cells.find(c=>!c.years['2024'])!
 await user.selectOptions(screen.getByLabelText('Inspect an area'),cell.id)
 await user.selectOptions(screen.getByLabelText('Period'),'2024')
 expect(screen.getByText('Select a recorded area')).toBeTruthy()
 expect(screen.queryByText('Observed severe share',{selector:'dt'})).toBeNull()
})
