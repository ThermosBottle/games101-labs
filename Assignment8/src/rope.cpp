#include <iostream>
#include <vector>

#include "CGL/vector2D.h"

#include "mass.h"
#include "rope.h"
#include "spring.h"

namespace CGL
{

    const double damping = 0.00005;
    Rope::Rope(Vector2D start, Vector2D end, int num_nodes, float node_mass, float k, vector<int> pinned_nodes)
    {
        // TODO (Part 1): Create a rope starting at `start`, ending at `end`, and containing `num_nodes` nodes.

        //        Comment-in this part when you implement the constructor
        //        for (auto &i : pinned_nodes) {
        //            masses[i]->pinned = true;
        //        }
        for (int i = 0; i < num_nodes; i++)
        {
            Vector2D position = start + (end - start) * (float(i) / float(num_nodes - 1));
            bool pinned = false;
            for (auto &j : pinned_nodes)
            {
                if (i == j)
                {
                    pinned = true;
                    break;
                }
            }
            Mass *mass = new Mass(position, node_mass, pinned);
            mass->last_position = position;
            masses.push_back(mass);
            if (i > 0)
            {
                Spring *spring = new Spring(masses[i - 1], masses[i], k);
                spring->rest_length = (masses[i - 1]->position - masses[i]->position).norm();
                springs.push_back(spring);
            }
        }
    }

    void Rope::simulateEuler(float delta_t, Vector2D gravity)
    {
        for (auto &s : springs)
        {
            // TODO (Part 2): Use Hooke's law to calculate the force on a node
            s->m1->forces += -s->k * (s->m1->position - s->m2->position).unit() * ((s->m1->position - s->m2->position).norm() - s->rest_length);
            s->m2->forces += -s->k * (s->m2->position - s->m1->position).unit() * ((s->m2->position - s->m1->position).norm() - s->rest_length);
        }

        for (auto &m : masses)
        {
            if (!m->pinned)
            {
                // TODO (Part 2): Add the force due to gravity, then compute the new velocity and position
                // semi-implicit Euler integration
                m->forces += gravity * m->mass;
                Vector2D acceleration = m->forces / m->mass;
                m->velocity += acceleration * delta_t;
                m->position += m->velocity * delta_t;

                // TODO (Part 2): Add global damping
                m->velocity *= (1.0f - damping);
            }

            // Reset all forces on each mass
            m->forces = Vector2D(0, 0);
        }
    }

    void Rope::simulateVerlet(float delta_t, Vector2D gravity)
    {
        for (auto &s : springs)
        {
            // TODO (Part 3): Simulate one timestep of the rope using explicit Verlet （solving constraints)
        }

        for (auto &m : masses)
        {
            if (!m->pinned)
            {
                Vector2D temp_position = m->position;
                // TODO (Part 3.1): Set the new position of the rope mass
                m->forces += gravity * m->mass;
                Vector2D acceleration = m->forces / m->mass;
                m->last_position = temp_position;

                // TODO (Part 4): Add global Verlet damping
                m->position = m->position + (1.0f - damping) * (m->position - m->last_position) + acceleration * delta_t * delta_t;
            }
        }
    }
}
