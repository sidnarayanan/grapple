#include "fastjet/PseudoJet.hh"
#include <vector>
#include <array>
#include <stdlib>
#include <time>

using namespace fastjet;
using namespace std;


template <int K, int N>
class HierarchicalOrdering {
public:
    HierarchicalOrdering() { }

    vector<vector<PseudoJet*>> 
    fit(vector<PseudoJet*> &particles)
    {
        return _recursive_fit(particles);
    }
    
private:
    vector<vector<PseudoJet*>> 
     _recursive_fit(vector<PseudoJet*> &particles)
    {
        vector<vector<PseudoJet*>> clusters;

        kmeans = KMeans<K>(particles);
        for (int i_k=0; i_k!=K; ++i_k) {
            auto& cluster = kmeans.get_clusters()[i_k];
            if (cluster.size() > N) {
                split_clusters = fit(cluster);
                for (auto& c : split_clusters) {
                    clusters.append(c);
                }
            } else {
                clusters.append(cluster);
            } 
        }
        return cluster;
    }

};


template<int K>
class KMeans { 
public:
    KMeans(vector<PseudoJet*> particles, int max_iter=20) 
    {
        // randomly initialize centroids
        array<int, K> i_centroids;
        for (int i=0; i!=K; ++i) {
            while (true) {
                auto i_p = rand() % particles.size();
                bool found = false;
                for (int j=0; j!=i; ++j) {
                    found = (i_centroids[j] == i_p);
                    if (found)
                        break;
                }
                if (!found) {
                    i_centroids[i] = i_p;
                    centroids[i][0] = particles[i_p]->Eta();
                    centroids[i][1] = particles[i_p]->Phi();
                    break;
                }
            }
        } 

        for (int i_iter=0; i_iter!=max_iter; ++i_iter) {
            assign_clusters(particles);
            update_centroids();
        }
    }

    ~KMeans() { }

    const array<vector<PseudoJet*>, K> get_clusters() { return clusters; }

private:
    array<array<float, 2>, K> centroids;
    array<vector<PseudoJet*>, K> clusters;

    void assign_particles(vector<PseudoJet*> &particles) 
    {
        for (int i=0; i!=K; ++i) {
            clusters[i].clear();
        }

        for (auto& p : particles) {
            float closest = 99999;
            int i_closest = -1;
            float eta = p->Eta(); float phi = p->Phi();

            for (int i=0; i!=K; ++i) {
                auto dr = (eta - clusters[i][0]) ** 2 + (phi - clusters[i][1]) ** 2;
                if (dr < closest) {
                    closest = dr;
                    i_closest = i;
                }
            }
            clusters[i_closest].push_back(&p);
        }
    }

    void update_centroids() 
    {
        for (int i=0; i!=K; ++i) {
            float eta_sum=0, phi_sum=0;
            auto &cluster = clusters[i];
            for (auto& p : cluster) {
                eta_sum += p->Eta();
                phi_sum += p->Phi();
            }
            centroids[i][0] = eta_sum / cluster.size();
            centroids[i][1] = phi_sum / cluster.size();
        }
    }
}; 
